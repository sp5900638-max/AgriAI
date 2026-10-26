# 🌾 AgriAI (CropSense AI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless-5C3EE8.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![SDG 2](https://img.shields.io/badge/SDG%202-Zero%20Hunger-DDA63A.svg)](https://sdgs.un.org/goals/goal2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Data-Driven Crop Health Analytics and Nutrient Deficiency Detection using Explainable Machine Learning and Computer Vision**  
> GitHub Repository: [sp5900638-max/AgriAI](https://github.com/sp5900638-max/AgriAI)

---

## 📌 Problem Statement

Crop diseases and nutrient deficiencies severely impair agricultural productivity, household food security, and smallholder farmers' livelihoods. Early in-field diagnosis is notoriously difficult because visible leaf symptoms often overlap across bacterial/fungal pathogens, nitrogen/potassium deficiencies, and environmental stress. Furthermore, agricultural telemetry is fragmented and farmers lack access to interpretable, multi-modal analytical tools.

**AgriAI** delivers an end-to-end, reproducible, and explainable decision-support system aligned with **United Nations Sustainable Development Goal 2 (Zero Hunger)**.

---

## 🌟 Key Features

1. **Computer Vision & Optical Quality Assurance (OpenCV)**:
   - **Quality Check**: Validates input images using Laplacian blur variance ($\ge 60$) and mean brightness ($40 - 230$), triggering an automated *"Retake Photo Prompt"* if quality criteria fail.
   - **Illumination Normalization**: CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to the LAB color space $L$-channel to handle harsh field sunlight and shadows.
   - **Segmentation & Lesion Quantification**: HSV foliage thresholding and contour detection quantifying **Lesion Area %**, **Lesion Count**, **Chlorosis Ratio**, and mapping to **Severity Tiers** (*Low*, *Medium*, *High*).
2. **Deep Learning Classifier & Grad-CAM Explainability (PyTorch)**:
   - Transfer learning using **MobileNetV2** with temperature-scaled calibrated probabilities.
   - Gradient-weighted Class Activation Mapping (**Grad-CAM**) highlighting the exact leaf spots and spatial features driving the neural network's diagnosis.
   - Out-of-distribution detection flagging ambiguous cases as *"Needs Expert Verification"*.
3. **Tabular Environmental Risk Modeling**:
   - Soil chemistry ($N, P, K, \text{pH}, \text{soil moisture}$) and meteorological data ($\text{temperature}, \text{relative humidity}, \text{precipitation}$).
   - **Random Forest** and **XGBoost** classifiers with 5-fold cross-validation and feature importance ranking.
   - Statistical hypothesis testing ($\chi^2$ independence test, One-Way ANOVA) and unsupervised pattern discovery (PCA + K-Means clustering).
4. **Bayesian Late Fusion with Graceful Fallback**:
   - Dynamically fuses visual CNN probabilities with environmental priors ($P_{fused} = \alpha P_{img} + (1-\alpha)P_{tab}$).
   - Automatically and gracefully falls back to image-only diagnostics when soil/weather sensors are unavailable.
5. **Interactive 6-Page Streamlit Dashboard**:
   - Multi-page UI featuring **Overview**, **Trends**, **Environment**, **Diagnosis (Interactive Multi-Modal Suite)**, **Model Performance**, and **Recommendations**.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Input_Layer ["Data Sources"]
        IMG["Leaf Image (Upload / Camera)"]
        TAB["Soil NPK, pH, Moisture + Weather Telemetry"]
    end

    subgraph CV_Pipeline ["Computer Vision Path (OpenCV & PyTorch)"]
        QC["Quality Check (Laplacian Blur & Brightness)"]
        PRE["Denoise & CLAHE Illumination Equalization"]
        SEG["HSV Foliage Segmentation"]
        LES["Contour Detection -> Lesion %, Count, Chlorosis"]
        CNN["MobileNetV2 Classifier + Temperature Scaling"]
        GCAM["Grad-CAM Explainability Heatmap"]
        
        IMG --> QC
        QC -->|Pass| PRE --> SEG --> LES
        SEG --> CNN
        CNN --> GCAM
    end

    subgraph Tabular_Pipeline ["Tabular Telemetry Path (Scikit-Learn & XGBoost)"]
        CLEAN["Imputation (KNN/Median) & Outlier Clipping (IQR)"]
        RF_XGB["Random Forest & XGBoost Classifiers (5-Fold CV)"]
        SHAP_IMP["Feature Importance Ranking & ANOVA Stats"]
        
        TAB --> CLEAN --> RF_XGB --> SHAP_IMP
    end

    subgraph Fusion_Layer ["Late Fusion Engine & Decision Support"]
        FUSE["Bayesian Late Fusion (Image Probs + Tabular Priors)"]
        FALLBACK["Graceful Fallback (Image-Only when Sensors Absent)"]
        UNCERT["Uncertainty Filter (Flag 'Needs Expert Verification')"]
        REC["Actionable Agronomic Interventions & Soil Amendments"]
        
        CNN --> FUSE
        RF_XGB --> FUSE
        CNN -.->|No Tabular Data| FALLBACK
        FUSE --> UNCERT --> REC
        FALLBACK --> UNCERT
    end

    subgraph UI_Layer ["Interactive Dashboard (Streamlit & Plotly)"]
        DASH["6-Page Dashboard (Overview, Trends, Environment, Diagnosis, Performance, Recs)"]
        REC --> DASH
        GCAM --> DASH
        LES --> DASH
    end
```

---

## 📁 Repository Structure

```text
sp5900638-max/AgriAI/
├── .gitignore                    # Excludes virtual environments, pycache, and logs
├── data/
│   ├── raw/
│   │   ├── images/               # Synthetic & benchmark leaf dataset across all 5 classes
│   │   └── tabular/              # Agricultural telemetry records (crop_health_records.csv)
│   └── processed/
├── notebooks/
│   ├── 01_eda_and_statistics.ipynb        # EDA, Chi-Square, ANOVA, PCA clustering
│   ├── 02_opencv_lesion_analysis.ipynb     # Image QA, CLAHE, HSV segmentation, lesion scoring
│   └── 03_model_training_and_eval.ipynb    # Tabular CV, CNN calibration, SVM baseline, Grad-CAM
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data cleaning, pHash duplicate detection, leakage checks
│   ├── opencv_utils.py           # Optical QA, CLAHE on LAB L-channel, HSV segmentation, lesion metrics
│   ├── features.py               # Handcrafted features (Color histograms, GLCM texture, shape)
│   ├── train_cnn.py              # PyTorch MobileNetV2 with temperature calibration, SVM baseline
│   ├── train_tabular.py          # RF & XGBoost (5-fold CV), Chi-Square, ANOVA, K-Means & PCA
│   ├── gradcam.py                # PyTorch Grad-CAM explainability hook & OpenCV heatmap overlay
│   └── fusion.py                 # Multi-modal late fusion with graceful vision-only fallback
├── models/                       # Serialized models (.pth, .pkl, JSON metadata)
├── reports/                      # Evaluation metrics, statistical summaries, ablation JSONs
├── dashboard/
│   └── app.py                    # 6-page interactive Streamlit & Plotly analytics dashboard
├── tests/
│   └── test_pipeline.py          # Pytest suite verifying full pipeline integrity (100% passing)
├── train_all.py                  # End-to-end execution runner
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Comprehensive documentation
```

---

## 📊 Benchmark Model Performance

| Model | Modality / Task | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | Tabular Risk Scoring | **94.17%** | 94.25% | 94.17% | **94.19%** |
| **Random Forest (5-Fold CV)** | Tabular Risk Scoring | **92.08%** | 92.15% | 92.08% | **92.11%** |
| **Classical SVM Baseline** | OpenCV Color + Texture + Shape | **96.25%** | 96.30% | 96.25% | **96.25%** |
| **CropSense CNN (MobileNetV2)**| Transfer Learning (Calibrated) | **75.00%** | 76.10% | 75.00% | **74.80%** |

### Top Environmental Predictors (Feature Importance):
1. **Potassium ($K$)**: `35.97%`
2. **Nitrogen ($N$)**: `35.85%`
3. **Relative Humidity**: `8.95%`
4. **Temperature**: `4.68%`
5. **Rainfall**: `4.42%`

---

## ⚡ Quickstart & Installation

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/sp5900638-max/AgriAI.git
cd AgriAI

# Create virtual environment with Python 3.12
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Integration Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
> Open your browser at `http://localhost:8501` to access the diagnostic dashboard.

### 4. Retrain Models & Regenerate Datasets
```bash
python train_all.py
```

---

## 🛡️ Robustness & Mitigation Strategies

1. **Lab-to-Field Domain Gap**: Lab datasets (like PlantVillage) feature uniform backgrounds, whereas field photos contain soil, weeds, and shadows. AgriAI utilizes CLAHE and HSV segmentation to isolate foliage and reports the lab vs. field performance gap openly in the dashboard.
2. **Calibrated Confidence**: Deep neural networks can be overconfident. AgriAI uses temperature scaling on softmax logits and calculates prediction entropy to flag uncertain classifications as *"Needs Expert Verification"*.
3. **Out-of-Distribution Rejection**: Images lacking clear leaf morphology or failing quality checks trigger an automatic retake prompt rather than hallucinating a false diagnosis.
4. **Decision Support Notice**: Outputs serve as evidence-backed guidance for farm scouting and extension prioritization, complementing rather than replacing certified laboratory pathology testing.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
