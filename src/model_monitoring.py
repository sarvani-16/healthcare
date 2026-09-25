"""
=============================================================================
VITALSIGN: EDUCATIONAL MODEL MONITORING & AUDIT (MLOPS)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Model: Random Forest Readmission Classifier
Outputs: outputs/Monitoring/monitoring_report.csv, monitoring_distribution.png
=============================================================================

ML Concept - Post-Deployment Model & Data Drift Monitoring:
In machine learning operations (MLOps), once a clinical model is deployed,
monitoring systems continuously inspect incoming inference streams for:
1. Data Drift: Shifts in patient demographic or physiological distributions.
2. Concept Drift: Evolution of clinical protocols or disease patterns over time.
3. Prediction Drift: Disproportionate inflation in positive prediction frequency.

ACADEMIC NOTE:
This script provides an educational demonstration of monitoring principles and
data auditing. It is not an active real-time hospital telemetry daemon.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
MODEL_PATH = BASE_DIR / "models" / "vitalsign_readmission_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs" / "Monitoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

DROP_COLUMNS = [
    'encounter_id', 'patient_nbr', 'readmitted',
    'weight', 'payer_code', 'medical_specialty'
]

def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)
    return df.replace("?", np.nan)

def main():
    print("=" * 70)
    print("VITALSIGN: EDUCATIONAL HEALTHCARE MODEL MONITORING AUDIT")
    print("=" * 70)

    # 1. Dataset Shape and Missing Values Audit
    df = load_data()
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100

    print("\n[1] DATA HEALTH & INTEGRITY AUDIT:")
    print(f"    - Total Hospital Encounters:   {total_rows:,}")
    print(f"    - Total Feature Attributes:    {total_cols}")
    print(f"    - Total Missing Values:        {missing_cells:,} ({missing_pct:.2f}% of all cells)")
    print(f"    - Duplicate Records:           {df.duplicated().sum()}")

    # 2. Target Distribution
    df['Readmission_30_Days'] = (df['readmitted'] == '<30').astype(int)
    target_counts = df['Readmission_30_Days'].value_counts()
    target_pct = df['Readmission_30_Days'].value_counts(normalize=True) * 100

    print("\n[2] TARGET CLASS DISTRIBUTION (GROUND TRUTH):")
    print(f"    - Class 0 (No Readmit / >30d): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
    print(f"    - Class 1 (Readmitted <30d):   {target_counts[1]:,} ({target_pct[1]:.2f}%)")

    # 3. Model Inference & Prediction Distribution
    if not MODEL_PATH.exists():
        print(f"\n[!] Model file {MODEL_PATH} not found. Running Random Forest training...")
        import subprocess, sys
        rf_script = Path(__file__).parent / "Random_Forest_Classifier_M3.py"
        subprocess.run([sys.executable, str(rf_script)], check=True)

    pipeline = joblib.load(MODEL_PATH)

    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns] + ['Readmission_30_Days'])
    y = df['Readmission_30_Days']

    # Stratified test split for monitoring evaluation
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    pred_counts = pd.Series(y_pred).value_counts()
    pred_pct = pd.Series(y_pred).value_counts(normalize=True) * 100

    print("\n[3] MODEL PREDICTION DISTRIBUTION (Holdout Batch of 10,000 Encounters):")
    print(f"    - Predicted Class 0 (Low Risk):  {pred_counts.get(0, 0):,} ({pred_pct.get(0, 0):.2f}%)")
    print(f"    - Predicted Class 1 (High Risk): {pred_counts.get(1, 0):,} ({pred_pct.get(1, 0):.2f}%)")
    print(f"    - Mean Predicted Probability:    {y_prob.mean():.4f} ({y_prob.mean() * 100:.2f}%)")

    # 4. Live Accuracy Verification
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n[4] VERIFIED BATCH ACCURACY METRICS:")
    print(f"    - Accuracy:  {acc:.4f} ({acc:.2%})")
    print(f"    - Precision: {prec:.4f}")
    print(f"    - Recall:    {rec:.4f} ({rec:.2%})")
    print(f"    - F1-Score:  {f1:.4f}")

    # 5. Save Monitoring Report CSV
    report_data = [
        {'Metric': 'Total Dataset Rows', 'Value': str(total_rows)},
        {'Metric': 'Total Dataset Columns', 'Value': str(total_cols)},
        {'Metric': 'Total Missing Cells', 'Value': str(missing_cells)},
        {'Metric': 'Missing Cell Percentage', 'Value': f"{missing_pct:.2f}%"},
        {'Metric': 'Ground Truth Class 0 Count', 'Value': str(target_counts[0])},
        {'Metric': 'Ground Truth Class 1 Count', 'Value': str(target_counts[1])},
        {'Metric': 'Ground Truth Readmission Rate', 'Value': f"{target_pct[1]:.2f}%"},
        {'Metric': 'Batch Predicted Class 0 Count', 'Value': str(pred_counts.get(0, 0))},
        {'Metric': 'Batch Predicted Class 1 Count', 'Value': str(pred_counts.get(1, 0))},
        {'Metric': 'Batch Mean Predicted Probability', 'Value': f"{y_prob.mean():.4f}"},
        {'Metric': 'Batch Test Accuracy', 'Value': f"{acc:.4f}"},
        {'Metric': 'Batch Test Precision', 'Value': f"{prec:.4f}"},
        {'Metric': 'Batch Test Recall', 'Value': f"{rec:.4f}"},
        {'Metric': 'Batch Test F1-Score', 'Value': f"{f1:.4f}"}
    ]

    report_df = pd.DataFrame(report_data)
    report_csv = OUTPUT_DIR / "monitoring_report.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"\n[OK] Monitoring audit report saved to: {report_csv}")

    # 6. Generate Monitoring Distribution Graph
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Actual vs Predicted Distribution
    categories = ['No Readmit (0)', 'Readmitted (1)']
    actual_shares = [target_pct[0], target_pct[1]]
    pred_shares = [pred_pct.get(0, 0), pred_pct.get(1, 0)]

    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width/2, actual_shares, width, label='Ground Truth Actual', color='#3b82f6')
    axes[0].bar(x + width/2, pred_shares, width, label='Model Predictions', color='#10b981')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=11)
    axes[0].set_ylabel('Percentage of Encounters (%)', fontsize=11)
    axes[0].set_title('Actual vs Predicted Target Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')

    # Probability Histogram
    axes[1].hist(y_prob, bins=25, color='#8b5cf6', edgecolor='black', alpha=0.7)
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
    axes[1].set_title('Predicted Probability Distribution (Holdout Batch)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Readmission Probability', fontsize=11)
    axes[1].set_ylabel('Number of Patients', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('VitalSign: Model Monitoring & Audit Telemetry', fontsize=14, fontweight='bold')
    plt.tight_layout()

    dist_png = OUTPUT_DIR / "monitoring_distribution.png"
    plt.savefig(dist_png, dpi=300)
    plt.close()
    print(f"[OK] Monitoring distribution plot saved to: {dist_png}")
    print("=" * 70)

if __name__ == "__main__":
    main()
