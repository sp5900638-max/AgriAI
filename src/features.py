"""
Feature Extraction Utility for CropSense AI
Extracts handcrafted computer vision features for classical ML baselines:
- Color histograms across HSV, LAB, and RGB color spaces
- GLCM (Gray-Level Co-occurrence Matrix) texture descriptors
- Morphological shape descriptors (aspect ratio, circularity, solidity, extent)
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import Dict, Any, List


def extract_color_histogram(image_bgr: np.ndarray, mask: np.ndarray = None, bins_per_channel: int = 16) -> np.ndarray:
    """
    Computes normalized color histograms across HSV, LAB, and BGR color spaces.
    """
    features = []
    
    # 1. HSV Histogram
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], mask, [bins_per_channel], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # 2. LAB Histogram (Luminance & Color Opponents)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    for i in range(3):
        hist = cv2.calcHist([lab], [i], mask, [bins_per_channel], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # 3. BGR Histogram
    for i in range(3):
        hist = cv2.calcHist([image_bgr], [i], mask, [bins_per_channel], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    return np.array(features, dtype=np.float32)


def extract_glcm_texture_features(image_bgr: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Computes Haralick GLCM texture features:
    Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Downsample gray levels to 32 for robust co-occurrence statistics
    gray_quantized = (gray // 8).astype(np.uint8)

    # Compute GLCM at distances [1, 3] and angles [0, 45, 90, 135 degrees]
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_quantized, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        val = graycoprops(glcm, prop)
        features.append(np.mean(val))
        features.append(np.std(val))

    return np.array(features, dtype=np.float32)


def extract_shape_descriptors(mask: np.ndarray) -> np.ndarray:
    """
    Computes morphological shape descriptors from the leaf / lesion binary mask.
    Features: Area ratio, Perimeter, Circularity, Aspect Ratio, Extent, Solidity.
    """
    total_area = float(mask.shape[0] * mask.shape[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(6, dtype=np.float32)

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    
    # Circularity (isoperimetric quotient)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Bounding rectangle properties
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    rect_area = float(w * h)
    extent = area / rect_area if rect_area > 0 else 0.0

    # Convex hull & solidity
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    features = [
        area / total_area,
        perimeter / (mask.shape[0] + mask.shape[1]),
        circularity,
        aspect_ratio,
        extent,
        solidity
    ]
    return np.array(features, dtype=np.float32)


def extract_all_opencv_features(image_bgr: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Extracts concatenated feature vector (Color + Texture + Shape)
    for training and testing classical ML baseline models (SVM, Random Forest).
    """
    if mask is None:
        # Default full image mask
        mask = np.ones((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8) * 255

    color_feats = extract_color_histogram(image_bgr, mask=mask)
    glcm_feats = extract_glcm_texture_features(image_bgr, mask=mask)
    shape_feats = extract_shape_descriptors(mask)

    return np.concatenate([color_feats, glcm_feats, shape_feats], axis=0)
