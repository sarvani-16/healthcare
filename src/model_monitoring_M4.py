# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/Monitoring"
MODELS_DIR = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/models"
MODEL_PATH = os.path.join(MODELS_DIR, "vitalsign_readmission_model.pkl")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
    'weight', 'payer_code', 'medical_specialty',
    'diag_1', 'diag_2', 'diag_3'
]

# ============================================================
# 4. LOAD DATASET
# ============================================================
print("=" * 60)
print("VITALSIGN - MODEL MONITORING & AUDITING (M4 / MLOps)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING & DATA INTEGRITY AUDIT
# ============================================================
total_rows, total_cols = data.shape
total_cells = total_rows * total_cols
missing_cells = (data == '?').sum().sum() + data.isnull().sum().sum()
missing_pct = (missing_cells / total_cells) * 100

data = data.replace("?", np.nan)
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

target_counts = data['Readmission_30_Days'].value_counts()
target_pct = data['Readmission_30_Days'].value_counts(normalize=True) * 100

print(f"\n[1] DATA INTEGRITY AUDIT:")
print(f"  - Total Encounters:         {total_rows:,}")
print(f"  - Total Columns:            {total_cols}")
print(f"  - Total Missing Values:     {missing_cells:,} ({missing_pct:.2f}% of all cells)")
print(f"  - Class 0 (No Readmit):     {target_counts[0]:,} ({target_pct[0]:.2f}%)")
print(f"  - Class 1 (Readmitted <30d):{target_counts[1]:,} ({target_pct[1]:.2f}%)")

# ============================================================
# 6. MODEL / ANALYSIS (BATCH PREDICTION ON HOLDOUT DATA)
# ============================================================
if not os.path.exists(MODEL_PATH):
    print("Training Champion Model before running Monitoring Audit...")
    import subprocess, sys
    rf_script = os.path.join(os.path.dirname(__file__), "Random_Forest_Classifier_M3.py")
    subprocess.run([sys.executable, rf_script], check=True)

pipeline = joblib.load(MODEL_PATH)

X = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns] + ['Readmission_30_Days'])
y = data['Readmission_30_Days']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nRunning batch inference on holdout test set ({len(X_test):,} encounters)...")
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

pred_counts = pd.Series(y_pred).value_counts()
pred_pct = pd.Series(y_pred).value_counts(normalize=True) * 100

# ============================================================
# 7. EVALUATION / DRIFT & ACCURACY METRICS
# ============================================================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n[2] BATCH PREDICTION DISTRIBUTION:")
print(f"  - Predicted Low Risk (0):  {pred_counts.get(0, 0):,} ({pred_pct.get(0, 0):.2f}%)")
print(f"  - Predicted High Risk (1): {pred_counts.get(1, 0):,} ({pred_pct.get(1, 0):.2f}%)")
print(f"  - Mean Predicted Probability: {y_prob.mean():.4f}")

print(f"\n[3] PERFORMANCE DRIFT AUDIT:")
print(f"  - Accuracy:  {acc:.4f} ({acc:.2%})")
print(f"  - Precision: {prec:.4f}")
print(f"  - Recall:    {rec:.4f} ({rec:.2%})")
print(f"  - F1-Score:  {f1:.4f}")

# ============================================================
# 8. SAVE RESULTS (REPORT CSV, DISTRIBUTION GRAPH, TEXT)
# ============================================================
# 1. Monitoring Report CSV
monitoring_records = [
    {'Metric': 'Total Dataset Rows', 'Value': str(total_rows)},
    {'Metric': 'Total Attributes', 'Value': str(total_cols)},
    {'Metric': 'Missing Cells Percentage', 'Value': f"{missing_pct:.2f}%"},
    {'Metric': 'Ground Truth Class 1 Rate', 'Value': f"{target_pct[1]:.2f}%"},
    {'Metric': 'Test Batch Encounters', 'Value': str(len(X_test))},
    {'Metric': 'Predicted High Risk Rate', 'Value': f"{pred_pct.get(1, 0):.2f}%"},
    {'Metric': 'Mean Readmission Probability', 'Value': f"{y_prob.mean():.4f}"},
    {'Metric': 'Test Accuracy', 'Value': f"{acc:.4f}"},
    {'Metric': 'Test Precision', 'Value': f"{prec:.4f}"},
    {'Metric': 'Test Recall', 'Value': f"{rec:.4f}"},
    {'Metric': 'Test F1-Score', 'Value': f"{f1:.4f}"}
]
report_df = pd.DataFrame(monitoring_records)
report_csv = os.path.join(OUTPUT_FOLDER, "monitoring_report.csv")
report_df.to_csv(report_csv, index=False)

# 2. Prediction Distribution Graph
plt.figure(figsize=(8, 5))
sns.histplot(y_prob, bins=30, kde=True, color='#2563eb')
plt.axvline(0.5, color='#dc2626', linestyle='--', linewidth=2, label='Decision Threshold (0.50)')
plt.title("Deployment Inference: Predicted Probability Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Probability of 30-Day Readmission", fontsize=10)
plt.ylabel("Number of Encounters", fontsize=10)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
dist_png = os.path.join(OUTPUT_FOLDER, "prediction_distribution.png")
plt.savefig(dist_png, dpi=300)
plt.close()

# 3. Text summary
summary_txt = os.path.join(OUTPUT_FOLDER, "monitoring_summary.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("VITALSIGN MODEL MONITORING AUDIT SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    for r in monitoring_records:
        f.write(f"{r['Metric']:<35}: {r['Value']}\n")

print(f"\n[OK] Monitoring CSV report saved to: {report_csv}")
print(f"[OK] Distribution graph saved to:    {dist_png}")
print(f"[OK] Summary text saved to:          {summary_txt}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: MODEL MONITORING AUDIT COMPLETED")
print("=" * 60)
