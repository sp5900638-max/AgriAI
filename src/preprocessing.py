"""
Data Cleaning, Preprocessing, and Leakage Verification for CropSense AI
Modules:
- Missing value imputation (Median/Mode & KNN Imputer)
- Outlier filtering (IQR & Z-score)
- Image perceptual hashing (imagehash) for duplicate detection
- Tabular standardization & categorical encoding
- Stratified Train/Val/Test split with leakage checks
- Benchmark dataset generation for end-to-end reproducibility
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from typing import Tuple, List, Dict, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer


SUPPORTED_CLASSES = [
    "Healthy",
    "Nitrogen_Deficiency",
    "Potassium_Deficiency",
    "Early_Blight",
    "Bacterial_Spot"
]

CROPS = ["Tomato", "Potato", "Corn", "Wheat", "Rice"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
LOCATIONS = ["Punjab", "Maharashtra", "Uttar Pradesh", "Karnataka", "Madhya Pradesh"]


def compute_image_hashes(image_paths: List[str]) -> Dict[str, str]:
    """
    Computes perceptual hashes (pHash) for image duplicate detection.
    """
    hashes = {}
    for p in image_paths:
        try:
            with Image.open(p) as img:
                h = str(imagehash.phash(img))
                hashes[p] = h
        except Exception:
            continue
    return hashes


def find_duplicate_images(hash_dict: Dict[str, str], threshold: int = 4) -> List[Tuple[str, str, int]]:
    """
    Finds near-duplicate images based on Hamming distance between perceptual hashes.
    """
    duplicates = []
    items = list(hash_dict.items())
    for i in range(len(items)):
        path_a, hash_str_a = items[i]
        h_a = imagehash.hex_to_hash(hash_str_a)
        for j in range(i + 1, len(items)):
            path_b, hash_str_b = items[j]
            h_b = imagehash.hex_to_hash(hash_str_b)
            distance = h_a - h_b
            if distance <= threshold:
                duplicates.append((path_a, path_b, distance))
    return duplicates


def handle_missing_values(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    strategy: str = "knn"
) -> pd.DataFrame:
    """
    Imputes missing values using Median/Mode or KNN Imputer.
    """
    df_clean = df.copy()

    # Impute categorical columns with Mode
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(mode_val)

    # Impute numerical columns
    if numerical_cols:
        if strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])
        else:
            imputer = SimpleImputer(strategy="median")
            df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])

    return df_clean


def remove_tabular_outliers(
    df: pd.DataFrame,
    numerical_cols: List[str],
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detects and caps/removes outliers using IQR or Z-score method.
    """
    df_clean = df.copy()
    
    if method == "iqr":
        for col in numerical_cols:
            if col in df_clean.columns:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                df_clean[col] = df_clean[col].clip(lower, upper)
    elif method == "zscore":
        for col in numerical_cols:
            if col in df_clean.columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 0:
                    lower = mean - threshold * std
                    upper = mean + threshold * std
                    df_clean[col] = df_clean[col].clip(lower, upper)
                    
    return df_clean


def check_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    identifier_cols: List[str]
) -> Dict[str, Any]:
    """
    Verifies that no identical or overlapping data exists across splits.
    """
    train_set = set(tuple(x) for x in train_df[identifier_cols].values)
    val_set = set(tuple(x) for x in val_df[identifier_cols].values)
    test_set = set(tuple(x) for x in test_df[identifier_cols].values)

    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    passed = (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0)

    return {
        "passed": passed,
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df)
    }


def split_data_stratified(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a 3-way stratified split: Train, Validation, and Test.
    """
    # First split off test set
    train_val, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )

    # Then split train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        stratify=train_val[target_col],
        random_state=random_state
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def generate_benchmark_datasets(
    output_dir: str = "data",
    num_samples: int = 1500,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Generates realistic, physically grounded benchmark agricultural data:
    1. Tabular soil, weather, crop records, and health risks
    2. Synthetic leaf images representing healthy, chlorotic, and necrotic lesions
    """
    np.random.seed(random_state)
    os.makedirs(os.path.join(output_dir, "raw", "tabular"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)

    records = []
    
    for i in range(num_samples):
        crop = np.random.choice(CROPS)
        season = np.random.choice(SEASONS)
        location = np.random.choice(LOCATIONS)
        
        # Environmental and Soil ranges
        # Realistic soil chemistry
        n = np.random.uniform(20.0, 180.0)    # Nitrogen (kg/ha)
        p = np.random.uniform(10.0, 90.0)     # Phosphorus (kg/ha)
        k = np.random.uniform(15.0, 120.0)    # Potassium (kg/ha)
        ph = np.random.uniform(5.0, 8.5)      # Soil pH
        soil_moisture = np.random.uniform(15.0, 85.0) # %
        
        # Weather
        temp = np.random.uniform(16.0, 38.0)  # Celsius
        humidity = np.random.uniform(30.0, 95.0) # %
        rainfall = np.random.uniform(10.0, 250.0) # mm
        
        # Ground-truth biological risk / condition logic
        if n < 45.0:
            disease_label = "Nitrogen_Deficiency"
            risk_level = "Medium"
        elif k < 30.0:
            disease_label = "Potassium_Deficiency"
            risk_level = "Medium"
        elif humidity > 80.0 and 22.0 <= temp <= 30.0 and rainfall > 120.0:
            disease_label = "Early_Blight"
            risk_level = "High"
        elif humidity > 75.0 and temp > 28.0 and soil_moisture > 65.0:
            disease_label = "Bacterial_Spot"
            risk_level = "High"
        else:
            disease_label = "Healthy"
            risk_level = "Low"

        # Occasional label noise / overlapping stress (realistic in agronomy)
        if np.random.rand() < 0.05:
            risk_level = np.random.choice(["Low", "Medium", "High"])

        img_filename = f"leaf_{i:04d}_{disease_label}.png"
        
        records.append({
            "sample_id": f"SMPL_{i:05d}",
            "crop_type": crop,
            "season": season,
            "location": location,
            "nitrogen": round(n, 2),
            "phosphorus": round(p, 2),
            "potassium": round(k, 2),
            "ph": round(ph, 2),
            "soil_moisture": round(soil_moisture, 2),
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
            "rainfall": round(rainfall, 2),
            "disease_label": disease_label,
            "risk_level": risk_level,
            "image_path": os.path.join(output_dir, "raw", "images", img_filename)
        })

    df = pd.DataFrame(records)
    tabular_path = os.path.join(output_dir, "raw", "tabular", "crop_health_records.csv")
    df.to_csv(tabular_path, index=False)

    # Generate synthetic leaf images for a representative subset (e.g. 100 images across classes)
    # to facilitate instant testing and dashboard demonstration
    img_dir = os.path.join(output_dir, "raw", "images")
    generate_synthetic_leaf_images(df.head(100), img_dir)

    return {
        "tabular_path": tabular_path,
        "images_dir": img_dir
    }


def generate_synthetic_leaf_images(sample_df: pd.DataFrame, output_dir: str):
    """
    Renders realistic synthetic leaf images with simulated chlorosis, necrosis, and spots
    matching OpenCV lesion segmentation expectations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in sample_df.iterrows():
        img_path = row["image_path"]
        disease = row["disease_label"]
        
        # 256x256 image with neutral background
        canvas = np.full((256, 256, 3), 45, dtype=np.uint8) # Dark neutral background
        
        # Draw central leaf ellipse (healthy green base: B=40, G=150, R=50)
        center = (128, 128)
        axes = (80, 110)
        angle = np.random.uniform(-15, 15)
        
        base_color = [35, 150, 45] # BGR Green
        cv2.ellipse(canvas, center, axes, angle, 0, 360, base_color, -1)
        
        # Add leaf veins
        for v in range(-60, 70, 25):
            pt1 = (128, 128 + v)
            pt2 = (128 + int(np.random.choice([-45, 45])), 128 + v - 10)
            cv2.line(canvas, pt1, pt2, (45, 175, 55), 2)
        cv2.line(canvas, (128, 20), (128, 235), (45, 180, 55), 3)

        # Apply specific visual symptoms per disease class
        if disease == "Nitrogen_Deficiency":
            # General chlorosis: Yellowing spread across the leaf blade (BGR: [30, 210, 225])
            overlay = canvas.copy()
            cv2.ellipse(overlay, center, (65, 95), angle, 0, 360, (20, 200, 220), -1)
            cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
            
        elif disease == "Potassium_Deficiency":
            # Marginal chlorosis / necrosis around outer margins (Brown/yellow tips)
            cv2.ellipse(canvas, center, axes, angle, 0, 360, (15, 130, 190), 8)
            # Add small burnt edge spots
            for _ in range(8):
                bx = int(128 + np.random.uniform(-70, 70))
                by = int(128 + np.random.uniform(-90, 90))
                cv2.circle(canvas, (bx, by), np.random.randint(4, 9), (15, 45, 120), -1)

        elif disease == "Early_Blight":
            # Dark brown concentric target-like spots
            for _ in range(np.random.randint(3, 7)):
                bx = int(128 + np.random.uniform(-40, 40))
                by = int(128 + np.random.uniform(-60, 60))
                r = np.random.randint(10, 22)
                # Outer chlorotic halo
                cv2.circle(canvas, (bx, by), r + 4, (30, 190, 210), -1)
                # Inner necrotic target ring
                cv2.circle(canvas, (bx, by), r, (15, 30, 80), -1)
                cv2.circle(canvas, (bx, by), max(r - 5, 2), (25, 45, 110), -1)

        elif disease == "Bacterial_Spot":
            # Numerous small water-soaked dark angular specks
            for _ in range(np.random.randint(15, 35)):
                bx = int(128 + np.random.uniform(-55, 55))
                by = int(128 + np.random.uniform(-75, 75))
                cv2.circle(canvas, (bx, by), np.random.randint(2, 6), (10, 20, 60), -1)

        # Save image
        cv2.imwrite(img_path, canvas)
