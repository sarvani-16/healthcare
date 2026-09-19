"""
=============================================================================
VitalSign / HealthcarePrediction
Module: healthcare_model_evaluation.py
Replaces and enhances M3 evaluation, metrics, ROC, and diagnostics.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

try:
    from .healthcare_dataset import load_dataset
    from .healthcare_preprocessing import prepare_features_target, load_preprocessor
except ImportError:
    from healthcare_dataset import load_dataset
    from healthcare_preprocessing import prepare_features_target, load_preprocessor

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_CHARTS_DIR = BASE_DIR / "static" / "charts"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.autolayout": True
})

def _save_plot(fig, filename: str):
    out_file = OUTPUTS_DIR / filename
    static_file = STATIC_CHARTS_DIR / filename
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    fig.savefig(static_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved evaluation chart: {filename}")
    return str(out_file)

def load_saved_model():
    """Loads champion model and metadata from models/."""
    model_path = MODELS_DIR / "vitalsign_readmission_model.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "vitalsign_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found in {MODELS_DIR}. Run training first.")

    meta_path = MODELS_DIR / "model_metadata.pkl"
    if not meta_path.exists():
        meta_path = MODELS_DIR / "vitalsign_metadata.pkl"

    model = joblib.load(model_path)
    metadata = joblib.load(meta_path) if meta_path.exists() else {}
    return model, metadata

def recreate_test_split(test_size: float = 0.20, random_state: int = 42):
    """Recreates the exact stratified test split."""
    df = load_dataset()
    X, y = prepare_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_test, y_test

def evaluate_model():
    """Executes evaluation on test split, generates classification report, confusion matrix, and ROC curve."""
    model, metadata = load_saved_model()
    preprocessor = load_preprocessor()
    X_test, y_test = recreate_test_split()

    print(f"Loaded production model: {type(model).__name__}")
    print(f"Evaluating on {X_test.shape[0]} holdout test records...")

    X_test_trans = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_trans)
    y_proba = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else None

    # 1. Classification report
    class_names = metadata.get("class_names", ["No Readmission / >30d", "Readmitted <30d"])
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_text)

    report_file = OUTPUTS_DIR / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("VITALSIGN HEALTHCARE READMISSION MODEL EVALUATION REPORT\n")
        f.write(f"Model: {type(model).__name__}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
    print(f"[OK] Saved classification report to: {report_file}")

    # 2. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=["Not Readmitted (<30d)", "Readmitted (<30d)"],
                yticklabels=["Not Readmitted (<30d)", "Readmitted (<30d)"])
    ax.set_title(f"Confusion Matrix ({type(model).__name__})", pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Clinical Label", fontweight="bold")
    ax.set_ylabel("True Clinical Label", fontweight="bold")
    _save_plot(fig, "confusion_matrix.png")

    # 3. ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.3f})")
        ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance Baseline")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", pad=15, fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        _save_plot(fig, "roc_curve.png")

    # 4. Feature importance
    if hasattr(model, "feature_importances_"):
        try:
            feat_names = preprocessor.get_feature_names_out()
        except Exception:
            feat_names = np.array([f"Feature_{i}" for i in range(len(model.feature_importances_))])

        clean_names = [f.split('__')[-1] for f in feat_names]
        fi_df = pd.DataFrame({
            'Feature': clean_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color="#0ea5e9")
        ax.set_title("Top 20 Predictive Clinical Features (Feature Weights)", pad=15, fontweight="bold")
        ax.set_xlabel("Relative Importance Weight", fontweight="bold")
        _save_plot(fig, "feature_importance.png")
    # 5. Save evaluation_metrics.csv
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

    eval_df = pd.DataFrame([{
        "Model": type(model).__name__,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(auc, 4),
        "Holdout_Test_Records": len(y_test)
    }])
    eval_csv = OUTPUTS_DIR / "evaluation_metrics.csv"
    eval_df.to_csv(eval_csv, index=False)
    print(f"[OK] Saved evaluation metrics to: {eval_csv}")

    print("Model evaluation diagnostics completed successfully.")

if __name__ == "__main__":
    print("=" * 70)
    print("VITALSIGN: healthcare_model_evaluation.py Execution")
    print("=" * 70)
    evaluate_model()
    print("=" * 70)
