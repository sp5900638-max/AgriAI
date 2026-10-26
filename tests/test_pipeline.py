"""
Comprehensive Integration and Unit Tests for CropSense AI
Tests:
- OpenCV image quality checks (blur, luminance)
- CLAHE illumination enhancement and HSV segmentation
- Lesion quantification and severity tier calculation
- Feature extraction (Color + Texture + Shape)
- Grad-CAM heatmap generation
- Late fusion logic with and without tabular inputs (fallback)
"""

import os
import cv2
import numpy as np
import pytest
import torch

from src.opencv_utils import (
    check_image_quality,
    preprocess_image,
    enhance_illumination_clahe,
    segment_leaf,
    analyze_lesions
)
from src.features import extract_all_opencv_features
from src.train_cnn import CropSenseCNN, predict_single_leaf, SUPPORTED_CLASSES
from src.gradcam import GradCAM
from src.fusion import late_fusion_predict, compute_environmental_priors


def test_image_quality_check():
    # 1. Sharp image test
    sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp_img[20:80, 20:80] = 180
    res = check_image_quality(sharp_img, blur_threshold=10.0, min_brightness=10.0)
    assert res["passed"] is True
    assert res["blur_metric"] > 10.0

    # 2. Too dark image test
    dark_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res_dark = check_image_quality(dark_img, min_brightness=30.0)
    assert res_dark["passed"] is False
    assert "too dark" in res_dark["reason"]


def test_opencv_segmentation_and_lesions():
    # Create synthetic leaf with spot
    canvas = np.full((128, 128, 3), 40, dtype=np.uint8)
    # Green leaf center
    cv2.ellipse(canvas, (64, 64), (40, 50), 0, 0, 360, (30, 160, 45), -1)
    # Yellow chlorotic spot
    cv2.circle(canvas, (64, 64), 10, (20, 210, 225), -1)

    clahe = enhance_illumination_clahe(canvas)
    seg, mask = segment_leaf(clahe)
    assert np.count_nonzero(mask) > 0

    lesion_stats = analyze_lesions(canvas, mask)
    assert lesion_stats["leaf_area_px"] > 0
    assert lesion_stats["lesion_percent"] >= 0.0
    assert lesion_stats["severity_level"] in ["Low", "Medium", "High"]


def test_feature_extraction():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:50, 10:50] = (40, 150, 60)
    feats = extract_all_opencv_features(img)
    # Check that feature vector has non-zero length and valid finite values
    assert len(feats) > 100
    assert not np.isnan(feats).any()


def test_gradcam_execution():
    cnn = CropSenseCNN(num_classes=5, pretrained=False)
    cnn.eval()
    gradcam = GradCAM(cnn, cnn.features[-1])

    dummy_input = torch.randn(1, 3, 224, 224)
    cam = gradcam.generate_cam(dummy_input, target_class=0)
    assert cam.shape == (7, 7) or len(cam.shape) == 2
    assert 0.0 <= np.min(cam) <= np.max(cam) <= 1.0


def test_late_fusion_and_fallback():
    # Mock image prediction
    img_preds = {
        "prediction": "Nitrogen_Deficiency",
        "confidence": 0.85,
        "status": "Confident",
        "probabilities": {
            "Healthy": 0.05,
            "Nitrogen_Deficiency": 0.80,
            "Potassium_Deficiency": 0.05,
            "Early_Blight": 0.05,
            "Bacterial_Spot": 0.05
        }
    }

    # 1. Fallback mode (no tabular data)
    fallback_res = late_fusion_predict(img_preds, tabular_features=None)
    assert fallback_res["final_diagnosis"] == "Nitrogen_Deficiency"
    assert "Fallback" in fallback_res["fusion_mode"]

    # 2. Fused mode with low Nitrogen telemetry
    telemetry = {
        "nitrogen": 25.0, # severely deficient
        "phosphorus": 40.0,
        "potassium": 60.0,
        "ph": 6.5,
        "soil_moisture": 45.0,
        "temperature": 25.0,
        "humidity": 55.0,
        "rainfall": 50.0
    }
    fused_res = late_fusion_predict(img_preds, tabular_features=telemetry)
    assert fused_res["final_diagnosis"] == "Nitrogen_Deficiency"
    assert "Late Fusion" in fused_res["fusion_mode"]
    assert len(fused_res["recommendations"]) > 0
