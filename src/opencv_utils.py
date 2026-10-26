"""
OpenCV Image Processing and Lesion Analysis Utility for CropSense AI
Covers:
- Image quality assessment (Laplacian blur variance, brightness checks)
- Color conversion and standardization
- CLAHE illumination enhancement on LAB L-channel
- Leaf segmentation (HSV vegetation thresholding + morphological filtering)
- Lesion extraction & contour analysis
- Quantitative severity metrics (lesion %, chlorosis ratio, severity tier)
- Visualizations & overlay generation
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any


def check_image_quality(
    image: np.ndarray,
    blur_threshold: float = 60.0,
    min_brightness: float = 40.0,
    max_brightness: float = 230.0,
) -> Dict[str, Any]:
    """
    Checks if input image satisfies optical quality requirements.
    Rejects blurred or poorly lit images (overexposed / underexposed).
    """
    if image is None or image.size == 0:
        return {
            "passed": False,
            "blur_metric": 0.0,
            "brightness": 0.0,
            "reason": "Invalid or empty image"
        }

    # Convert to grayscale for metric calculations
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Blur detection via Laplacian variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    passed = True
    reasons = []

    if laplacian_var < blur_threshold:
        passed = False
        reasons.append(f"Image is blurry (Laplacian variance {laplacian_var:.1f} < {blur_threshold})")

    if brightness < min_brightness:
        passed = False
        reasons.append(f"Image is too dark (mean brightness {brightness:.1f} < {min_brightness})")
    elif brightness > max_brightness:
        passed = False
        reasons.append(f"Image is overexposed (mean brightness {brightness:.1f} > {max_brightness})")

    return {
        "passed": passed,
        "blur_metric": round(laplacian_var, 2),
        "brightness": round(brightness, 2),
        "reason": "; ".join(reasons) if not passed else "Optimal quality"
    }


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    denoise: bool = True
) -> np.ndarray:
    """
    Standardize dimensions, apply bilateral/Gaussian filter to reduce noise
    while preserving edge boundaries.
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    if denoise:
        # Bilateral filter smoothens textures while preserving leaf edges
        denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=50, sigmaSpace=50)
        return denoised
    return resized


def enhance_illumination_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB L-channel
    to normalize uneven sunlight/shadows across the leaf surface.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def segment_leaf(
    image_bgr: np.ndarray,
    use_morphology: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolate leaf body from background using HSV thresholding and morphology.
    Returns:
        segmented_bgr: BGR leaf image with background blacked out
        leaf_mask: Single channel binary mask (255 for leaf, 0 for background)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Vegetation color spectrum (covers healthy green, pale yellow, diseased brown)
    # Range 1: Green foliage
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Range 2: Chlorotic yellows & necrosis / brownish leaf parts
    lower_yellow_brown = np.array([10, 30, 30])
    upper_yellow_brown = np.array([25, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow_brown, upper_yellow_brown)

    # Combine vegetation masks
    combined_mask = cv2.bitwise_or(mask_green, mask_yellow)

    # Otsu thresholding on saturation/value as secondary support
    _, s_mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leaf_mask = cv2.bitwise_and(combined_mask, s_mask)

    if use_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Close holes inside leaf
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Remove small background specks
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find largest external contour (the main leaf)
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(leaf_mask)
        cv2.drawContours(clean_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
        leaf_mask = clean_mask

    segmented_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)
    return segmented_bgr, leaf_mask


def analyze_lesions(
    image_bgr: np.ndarray,
    leaf_mask: np.ndarray,
    min_lesion_area: int = 15
) -> Dict[str, Any]:
    """
    Perform contour detection and quantitative lesion analysis:
    - Lesion area %
    - Lesion count
    - Chlorosis ratio
    - Severity classification (Low, Medium, High)
    """
    leaf_pixel_count = int(np.count_nonzero(leaf_mask))
    if leaf_pixel_count == 0:
        return {
            "leaf_area_px": 0,
            "lesion_area_px": 0,
            "lesion_percent": 0.0,
            "lesion_count": 0,
            "chlorosis_ratio": 0.0,
            "severity_level": "None (No Leaf Detected)",
            "lesion_mask": np.zeros_like(leaf_mask),
            "contour_overlay": image_bgr.copy()
        }

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # Detect chlorosis (yellowing/nutrient deficiency): Hue 18-35 in HSV
    lower_chlorosis = np.array([18, 40, 40])
    upper_chlorosis = np.array([35, 255, 255])
    chlorosis_mask = cv2.inRange(hsv, lower_chlorosis, upper_chlorosis)
    chlorosis_mask = cv2.bitwise_and(chlorosis_mask, leaf_mask)
    chlorosis_pixel_count = int(np.count_nonzero(chlorosis_mask))

    # Detect necrotic lesions (dark brown, rust, black spots):
    # Low Value in HSV or distinct 'b' channel deviations in LAB
    lower_necrosis = np.array([0, 40, 20])
    upper_necrosis = np.array([18, 255, 170])
    necrosis_mask = cv2.inRange(hsv, lower_necrosis, upper_necrosis)
    necrosis_mask = cv2.bitwise_and(necrosis_mask, leaf_mask)

    # Combined diseased / lesion mask
    lesion_mask = cv2.bitwise_or(chlorosis_mask, necrosis_mask)

    # Morphological cleaning of tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

    # Find lesion contours
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_lesion_area]
    lesion_count = len(valid_contours)
    
    # Recalculate true lesion area from valid contours
    contour_mask = np.zeros_like(lesion_mask)
    cv2.drawContours(contour_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
    lesion_pixel_count = int(np.count_nonzero(contour_mask))

    lesion_percent = float((lesion_pixel_count / max(leaf_pixel_count, 1)) * 100.0)
    chlorosis_ratio = float(chlorosis_pixel_count / max(leaf_pixel_count, 1))

    # Determine Severity Level based on Section 5.4 definition:
    # severity = lesion area ÷ leaf area, then mapped to Low / Medium / High
    if lesion_percent < 10.0:
        severity_level = "Low"
    elif lesion_percent < 25.0:
        severity_level = "Medium"
    else:
        severity_level = "High"

    # Create visualization with overlaid contours
    overlay = image_bgr.copy()
    cv2.drawContours(overlay, valid_contours, -1, (0, 0, 255), 2)  # Red outline for lesions

    return {
        "leaf_area_px": leaf_pixel_count,
        "lesion_area_px": lesion_pixel_count,
        "lesion_percent": round(lesion_percent, 2),
        "lesion_count": lesion_count,
        "chlorosis_ratio": round(chlorosis_ratio, 3),
        "severity_level": severity_level,
        "lesion_mask": contour_mask,
        "contour_overlay": overlay
    }


def create_heatmap_overlay(
    image_bgr: np.ndarray,
    heatmap_grayscale: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Generates a color-mapped heatmap (e.g. Grad-CAM) and blends it with the original image.
    Uses cv2.applyColorMap and cv2.addWeighted as requested in Section 5.4.
    """
    # Normalize grayscale heatmap to [0, 255]
    if heatmap_grayscale.max() > 1.0:
        norm_map = np.uint8(np.clip(heatmap_grayscale, 0, 255))
    else:
        norm_map = np.uint8(np.clip(heatmap_grayscale * 255, 0, 255))

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(norm_map, (image_bgr.shape[1], image_bgr.shape[0]))
    
    # Apply color map
    color_map = cv2.applyColorMap(heatmap_resized, colormap)
    
    # Alpha blend
    blended = cv2.addWeighted(color_map, alpha, image_bgr, 1 - alpha, 0)
    return blended
