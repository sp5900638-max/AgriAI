"""
End-to-End Pipeline Runner for CropSense AI
Generates benchmark datasets, runs statistical tests, trains tabular models,
trains CNN & SVM baselines, and generates reports.
"""

import os
import json
from src.preprocessing import generate_benchmark_datasets
from src.train_tabular import train_and_evaluate_tabular_models, run_statistical_analysis, discover_patterns
from src.train_cnn import train_cnn_model
import pandas as pd


def main():
    print("=" * 60)
    print("CropSense AI: End-to-End Pipeline Execution")
    print("=" * 60)

    # 1. Generate benchmark datasets
    print("\n[Step 1/4] Generating Benchmark Agricultural Datasets & Images...")
    data_paths = generate_benchmark_datasets(output_dir="data", num_samples=1200)
    print(f"-> Tabular dataset: {data_paths['tabular_path']}")
    print(f"-> Leaf images directory: {data_paths['images_dir']}")

    # 2. Statistical Analysis & Pattern Discovery
    print("\n[Step 2/4] Running Statistical Analysis & EDA Patterns...")
    df = pd.read_csv(data_paths["tabular_path"])
    stats_results = run_statistical_analysis(df)
    patterns = discover_patterns(df)

    os.makedirs("reports", exist_ok=True)
    with open("reports/statistical_analysis.json", "w") as f:
        json.dump(stats_results, f, indent=2)
    with open("reports/patterns_pca_clusters.json", "w") as f:
        json.dump(patterns, f, indent=2)
    print("-> Statistical hypothesis tests & PCA clusters saved to reports/")

    # 3. Train Tabular Risk Models (Random Forest & XGBoost)
    print("\n[Step 3/4] Training Tabular Risk Models (RF + XGBoost) with 5-Fold CV...")
    tabular_results = train_and_evaluate_tabular_models(
        data_path=data_paths["tabular_path"],
        models_dir="models"
    )
    print(f"-> RF Accuracy: {tabular_results['rf_metrics']['accuracy'] * 100:.2f}%")
    print(f"-> XGBoost Accuracy: {tabular_results['xgb_metrics']['accuracy'] * 100:.2f}%")
    print("-> Feature Importances:", tabular_results["feature_importances"])

    with open("reports/tabular_evaluation.json", "w") as f:
        json.dump(tabular_results, f, indent=2)

    # 4. Train CNN Image Classifier & SVM Baseline
    print("\n[Step 4/4] Training CNN (MobileNetV2) & Classical SVM Baseline...")
    cnn_results = train_cnn_model(
        metadata_path=data_paths["tabular_path"],
        models_dir="models",
        epochs=3,
        batch_size=16
    )
    print(f"-> CNN Accuracy: {cnn_results['cnn_metrics']['accuracy'] * 100:.2f}%")
    print(f"-> Baseline SVM Accuracy: {cnn_results['svm_baseline_metrics']['accuracy'] * 100:.2f}%")

    with open("reports/image_evaluation.json", "w") as f:
        json.dump(cnn_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Pipeline Execution Completed Successfully!")
    print("Models saved in: models/")
    print("Reports saved in: reports/")
    print("=" * 60)


if __name__ == "__main__":
    main()
