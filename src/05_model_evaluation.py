"""
=============================================================================
VitalSign / HealthcarePrediction
Module 05: Model Evaluation & Performance Analysis
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_CHARTS_DIR = BASE_DIR / "static" / "charts"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_NAMES = [
    'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
    'glyburide_metformin', 'glipizide_metformin', 'glimepiride_pioglitazone',
    'metformin_rosiglitazone', 'metformin_pioglitazone', 'change', 'diabetesMed',
    'readmitted'
]

COLUMNS_TO_REMOVE = [
    'encounter_id',
    'patient_nbr',
    'weight',
    'payer_code',
    'medical_specialty',
    'readmitted'
]

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.autolayout": True
})

def load_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    df = df.replace("?", np.nan)
    return df

def save_chart(fig, filename: str):
    out_file = OUTPUTS_DIR / filename
    static_file = STATIC_CHARTS_DIR / filename
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    fig.savefig(static_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved evaluation chart: {filename}")

def main():
    print("=" * 70)
    print("VITALSIGN: 05_MODEL_EVALUATION")
    print("=" * 70)

    # 1. Load saved model, preprocessor, and metadata
    model_path = MODELS_DIR / "vitalsign_model.pkl"
    preprocessor_path = MODELS_DIR / "vitalsign_preprocessor.pkl"
    metadata_path = MODELS_DIR / "vitalsign_metadata.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("Saved model or preprocessor missing. Run 04_model_training.py first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}

    print(f"Loaded production model: {type(model).__name__}")

    # 2. Prepare test dataset (stratified 20%)
    df = load_data(DATASET_PATH)
    target_name = "Readmission_30_Days"
    df[target_name] = (
        df["readmitted"]
        .astype(str)
        .str.strip()
        .eq("<30")
        .astype(int)
    )

    cols_to_drop = [c for c in COLUMNS_TO_REMOVE + [target_name] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_name]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Transforming holdout test set ({X_test.shape[0]} records)...")
    X_test_trans = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_trans)
    y_proba = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else None

    # 3. Generate and save classification report (outputs/classification_report.txt)
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["No Readmission / >30d (Class 0)", "Readmitted <30d (Class 1)"]
    )
    print("\n--- Classification Report ---")
    print(report_text)

    report_file = OUTPUTS_DIR / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("VITALSIGN HEALTHCARE READMISSION PREDICTION REPORT\n")
        f.write(f"Model: {type(model).__name__}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
    print(f"[OK] Saved classification report to: {report_file}")

    # 4. Generate and save confusion matrix (outputs/confusion_matrix.png)
    print("Generating Confusion Matrix Plot...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=["Not Readmitted (<30d)", "Readmitted (<30d)"],
                yticklabels=["Not Readmitted (<30d)", "Readmitted (<30d)"])
    ax.set_title(f"Confusion Matrix ({type(model).__name__})", pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Clinical Label", fontweight="bold")
    ax.set_ylabel("True Clinical Label", fontweight="bold")
    save_chart(fig, "confusion_matrix.png")

    # 5. Generate and save ROC curve (outputs/roc_curve.png)
    if y_proba is not None:
        print("Generating ROC Curve Plot...")
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
        save_chart(fig, "roc_curve.png")

    # 6. Generate and save feature importance chart (outputs/feature_importance.png)
    if hasattr(model, "feature_importances_"):
        print("Generating Feature Importance Chart...")
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
        ax.set_xlabel("Relative Importance Score", fontweight="bold")
        save_chart(fig, "feature_importance.png")
    else:
        print(f"[INFO] Model {type(model).__name__} does not have tree feature_importances_. Using permutation importance placeholder.")

    print("\nModel evaluation completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
