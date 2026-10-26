"""
Tabular Risk Modeling, Statistical Analysis, and Pattern Discovery for CropSense AI
Modules:
- Statistical hypothesis testing (Chi-square test, ANOVA, Pearson/Spearman correlation)
- Pattern discovery (K-Means clustering, PCA)
- Random Forest and XGBoost model training with Stratified K-Fold CV
- Feature importance analysis
- Model persistence to `models/`
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import (
    handle_missing_values,
    remove_tabular_outliers,
    split_data_stratified,
    check_data_leakage
)


FEATURE_COLS = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "ph",
    "soil_moisture",
    "temperature",
    "humidity",
    "rainfall"
]

TARGET_COL = "risk_level"


def run_statistical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes statistical tests as per Section 5.3:
    - Pearson and Spearman correlations
    - Chi-square test (Category vs Season)
    - One-way ANOVA (Environmental variables across Health Risk levels)
    """
    results = {}

    # 1. Pearson & Spearman correlation matrices
    num_df = df[FEATURE_COLS]
    results["pearson_corr"] = num_df.corr(method="pearson").round(3).to_dict()
    results["spearman_corr"] = num_df.corr(method="spearman").round(3).to_dict()

    # 2. Chi-square test of independence: disease_label vs season
    contingency = pd.crosstab(df["disease_label"], df["season"])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    results["chi_square"] = {
        "chi2_stat": round(float(chi2), 3),
        "p_value": float(p_val),
        "dof": int(dof),
        "significant": bool(p_val < 0.05)
    }

    # 3. One-Way ANOVA across Risk Levels for key environmental factors
    anova_results = {}
    for col in FEATURE_COLS:
        groups = [group[col].dropna().values for _, group in df.groupby("risk_level")]
        if len(groups) > 1:
            f_stat, p = stats.f_oneway(*groups)
            anova_results[col] = {
                "f_stat": round(float(f_stat), 3),
                "p_value": float(p),
                "significant": bool(p < 0.05)
            }
    results["anova"] = anova_results

    return results


def discover_patterns(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Unsupervised pattern discovery:
    - PCA dimensionality reduction
    - K-Means clustering on environmental & soil vectors
    """
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    return {
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_coords": pca_coords.tolist(),
        "cluster_labels": cluster_labels.tolist(),
        "cluster_centers": kmeans.cluster_centers_.tolist()
    }


def train_and_evaluate_tabular_models(
    data_path: str = "data/raw/tabular/crop_health_records.csv",
    models_dir: str = "models",
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Cleans tabular data, trains Random Forest & XGBoost with Stratified K-Fold CV,
    evaluates performance, computes feature importances, and saves models.
    """
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Preprocessing
    df = handle_missing_values(df, numerical_cols=FEATURE_COLS, categorical_cols=["crop_type", "season", "location"])
    df = remove_tabular_outliers(df, numerical_cols=FEATURE_COLS, method="iqr")

    # Stratified Split
    train_df, val_df, test_df = split_data_stratified(df, target_col=TARGET_COL)
    
    # Leakage check
    leakage = check_data_leakage(train_df, val_df, test_df, identifier_cols=["sample_id"])

    # Encoders & Scalers
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[TARGET_COL])
    y_val = label_encoder.transform(val_df[TARGET_COL])
    y_test = label_encoder.transform(test_df[TARGET_COL])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_val = scaler.transform(val_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)

    # 2. XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, eval_metric="mlogloss")
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)

    # Cross Validation on Full Train+Val
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    cv_scores = []
    for train_idx, val_idx in skf.split(X_all, y_all):
        fold_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        fold_model.fit(X_all[train_idx], y_all[train_idx])
        fold_preds = fold_model.predict(X_all[val_idx])
        cv_scores.append(accuracy_score(y_all[val_idx], fold_preds))

    # Metrics
    rf_metrics = {
        "accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
        "precision": round(float(precision_score(y_test, rf_preds, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, rf_preds, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, rf_preds, average="weighted", zero_division=0)), 4),
        "cv_mean_accuracy": round(float(np.mean(cv_scores)), 4),
        "cv_std": round(float(np.std(cv_scores)), 4),
        "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist()
    }

    xgb_metrics = {
        "accuracy": round(float(accuracy_score(y_test, xgb_preds)), 4),
        "precision": round(float(precision_score(y_test, xgb_preds, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, xgb_preds, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, xgb_preds, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, xgb_preds).tolist()
    }

    # Feature Importance
    feature_importances = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, rf.feature_importances_)
    }
    # Sort descending
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    # Save artifacts
    joblib.dump(rf, os.path.join(models_dir, "tabular_rf_model.pkl"))
    joblib.dump(xgb, os.path.join(models_dir, "tabular_xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "tabular_scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "tabular_encoder.pkl"))

    return {
        "leakage_check": leakage,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "feature_importances": feature_importances,
        "target_classes": label_encoder.classes_.tolist()
    }


if __name__ == "__main__":
    results = train_and_evaluate_tabular_models()
    print("Tabular Models Trained Successfully:")
    print("Random Forest Accuracy:", results["rf_metrics"]["accuracy"])
    print("XGBoost Accuracy:", results["xgb_metrics"]["accuracy"])
    print("Top Feature Importances:", results["feature_importances"])