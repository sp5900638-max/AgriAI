"""
Late Fusion Engine and Agronomic Decision Support for CropSense AI
Modules:
- Late fusion combining CNN image probabilities with tabular environmental risk
- Graceful fallback when environmental data is unavailable
- Calibrated confidence validation
- Data-supported agronomic action plans and recommendations
- Safety and human-in-the-loop expert escalation checks
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple


# Agronomic knowledge mapping for soil/weather priors
DISEASE_ENVIRONMENT_MAPPING = {
    "Nitrogen_Deficiency": {
        "primary_trigger": "Low Soil Nitrogen (< 50 kg/ha)",
        "aggravators": ["Waterlogged soils", "High rainfall leaching"],
        "recommendations": [
            "Apply split dosage of Nitrogen fertilizer (e.g., Urea or Ammonium Sulfate) based on soil test.",
            "Incorporate well-decomposed organic manure or compost to enhance nitrogen retention.",
            "Ensure proper soil aeration and avoid prolonged waterlogging which hinders nitrogen absorption."
        ]
    },
    "Potassium_Deficiency": {
        "primary_trigger": "Low Soil Potassium (< 30 kg/ha)",
        "aggravators": ["Sandy soils with low cation exchange", "Acidic soil pH"],
        "recommendations": [
            "Top-dress with Muriate of Potash (MOP) or Potassium Sulfate (SOP) at recommended rates.",
            "Monitor soil pH; apply agricultural lime if pH < 5.5 to improve potassium availability.",
            "Avoid excessive calcium or magnesium applications that can antagonize potassium uptake."
        ]
    },
    "Early_Blight": {
        "primary_trigger": "Warm temperatures (22-30°C) with prolonged leaf wetness & high humidity (> 75%)",
        "aggravators": ["Overhead sprinkler irrigation", "Dense canopy without air circulation"],
        "recommendations": [
            "Apply preventative fungicides (e.g., Chlorothalonil, Mancozeb, or Copper Oxychloride).",
            "Switch from overhead sprinkler to drip irrigation to keep leaf foliage dry.",
            "Prune infected lower leaves and burn or bury infected plant residue away from the field.",
            "Enforce a 2-3 year crop rotation with non-solanaceous crops."
        ]
    },
    "Bacterial_Spot": {
        "primary_trigger": "Warm, humid conditions (> 75% RH, > 28°C) with rain splashing",
        "aggravators": ["Surface runoff", "Contaminated farming tools", "Insect feeding wounds"],
        "recommendations": [
            "Apply copper-based bactericides combined with Mancozeb to reduce resistant bacterial strains.",
            "Sanitize pruning tools and machinery between rows using a 10% bleach or alcohol solution.",
            "Avoid cultivating or moving through the crop canopy when leaves are wet to prevent bacterial spread.",
            "Utilize certified disease-free seeds or resistant crop cultivars."
        ]
    },
    "Healthy": {
        "primary_trigger": "Balanced soil nutrition and optimal climatic conditions",
        "aggravators": [],
        "recommendations": [
            "Continue regular monitoring and scout field margins twice weekly.",
            "Maintain balanced NPK fertilization schedule according to crop growth stage.",
            "Ensure adequate soil moisture through scheduled irrigation."
        ]
    }
}


def compute_environmental_priors(tabular_features: Dict[str, float]) -> Dict[str, float]:
    """
    Computes prior disease likelihoods from tabular soil and weather measurements.
    """
    n = tabular_features.get("nitrogen", 100.0)
    p = tabular_features.get("phosphorus", 40.0)
    k = tabular_features.get("potassium", 50.0)
    ph = tabular_features.get("ph", 6.5)
    moisture = tabular_features.get("soil_moisture", 50.0)
    temp = tabular_features.get("temperature", 25.0)
    humidity = tabular_features.get("humidity", 60.0)
    rainfall = tabular_features.get("rainfall", 50.0)

    # Initial uniform priors
    priors = {
        "Healthy": 0.2,
        "Nitrogen_Deficiency": 0.2,
        "Potassium_Deficiency": 0.2,
        "Early_Blight": 0.2,
        "Bacterial_Spot": 0.2
    }

    # Nutrient deficiency signals
    if n < 45.0:
        priors["Nitrogen_Deficiency"] += 1.5 * (1.0 - n / 45.0)
    if k < 30.0:
        priors["Potassium_Deficiency"] += 1.5 * (1.0 - k / 30.0)

    # Pathogen conditions
    if humidity > 75.0 and 20.0 <= temp <= 32.0 and rainfall > 100.0:
        priors["Early_Blight"] += 1.8 * ((humidity - 70.0) / 30.0)
    if humidity > 70.0 and temp > 27.0 and moisture > 60.0:
        priors["Bacterial_Spot"] += 1.6 * ((temp - 25.0) / 15.0)

    # Normal conditions favor healthy
    if 50.0 <= n <= 140.0 and 30.0 <= k <= 90.0 and humidity < 70.0:
        priors["Healthy"] += 1.2

    # Normalize to probabilities
    total = sum(priors.values())
    return {k_cls: round(v / total, 4) for k_cls, v in priors.items()}


def late_fusion_predict(
    image_predictions: Dict[str, Any],
    tabular_features: Optional[Dict[str, float]] = None,
    tabular_risk_output: Optional[Dict[str, Any]] = None,
    alpha: float = 0.70
) -> Dict[str, Any]:
    """
    Executes late fusion of visual CNN probabilities and tabular soil/weather priors.
    Includes graceful fallback to pure image prediction if tabular features are absent.
    """
    image_probs = image_predictions.get("probabilities", {})
    if not image_probs:
        # If image was rejected by quality check
        return {
            "final_diagnosis": image_predictions.get("prediction", "Rejected"),
            "confidence": 0.0,
            "fusion_mode": "Failed_Quality_Check",
            "status": image_predictions.get("status", "Invalid"),
            "fused_probabilities": {},
            "recommendations": ["Retake photo under clean lighting following quality guidelines."]
        }

    # GRACEFUL FALLBACK (Section 10 Mitigation):
    if tabular_features is None:
        top_class = image_predictions["prediction"]
        top_conf = image_predictions["confidence"]
        return {
            "final_diagnosis": top_class,
            "confidence": top_conf,
            "fusion_mode": "Image_Only (Fallback - No Tabular Data)",
            "status": image_predictions.get("status", "Confident"),
            "fused_probabilities": image_probs,
            "recommendations": DISEASE_ENVIRONMENT_MAPPING.get(top_class, {}).get("recommendations", []),
            "disclaimer": "Diagnosis generated from optical inspection alone. Soil and weather fusion recommended for field confirmation."
        }

    # Tabular Prior Computation
    env_priors = compute_environmental_priors(tabular_features)

    # Late Fusion Bayesian Weighting
    fused_probs = {}
    for cls in image_probs.keys():
        p_img = image_probs.get(cls, 0.0)
        p_tab = env_priors.get(cls, 0.2)
        fused_probs[cls] = alpha * p_img + (1.0 - alpha) * p_tab

    # Normalize
    total_fused = sum(fused_probs.values())
    fused_probs = {k: round(v / total_fused, 4) for k, v in fused_probs.items()}

    # Final top diagnosis
    sorted_classes = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
    final_diag, final_conf = sorted_classes[0]

    # Uncertainty check
    if final_conf < 0.55:
        status = "Needs Expert Verification"
    else:
        status = "Confident (Multi-Modal Fused)"

    # Compile actionable recommendations
    base_recs = DISEASE_ENVIRONMENT_MAPPING.get(final_diag, {}).get("recommendations", [])
    custom_recs = list(base_recs)

    # Tabular risk alerts
    if tabular_risk_output and tabular_risk_output.get("predicted_risk") == "High":
        custom_recs.insert(0, "URGENT: Environmental telemetry indicates high stress risk. Immediate scouting advised.")

    return {
        "final_diagnosis": final_diag,
        "confidence": round(final_conf, 4),
        "fusion_mode": "Multi-Modal Late Fusion (Vision + Soil/Weather)",
        "status": status,
        "top_3": sorted_classes[:3],
        "fused_probabilities": fused_probs,
        "environmental_priors": env_priors,
        "recommendations": custom_recs,
        "disclaimer": (
            "NOTICE: CropSense AI provides decision-support analytics under SDG 2 (Zero Hunger). "
            "These findings should guide field investigation and not replace laboratory soil testing "
            "or certified agricultural extension certification."
        )
    }
