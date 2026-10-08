# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M1_Dataset_EDA"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

# ============================================================
# 4. LOAD DATASET
# ============================================================
print("=" * 60)
print("VITALSIGN - CORRELATION MATRIX & BOXPLOT ANALYSIS (M1)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING
# ============================================================
data = data.replace("?", np.nan)
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

# ============================================================
# 6. MODEL / ANALYSIS (CORRELATION MATRIX)
# ============================================================
num_cols = [
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'Readmission_30_Days'
]

corr_matrix = data[num_cols].corr()
print("\n--- PEARSON CORRELATION MATRIX ---")
print(corr_matrix.round(3))

# ============================================================
# 7. EVALUATION / VISUALIZATION
# ============================================================
# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Numerical Features Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_FOLDER, "Correlation_Heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

# 2. Boxplots of Numerical Features vs Target
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
key_features = ['time_in_hospital', 'num_medications', 'number_inpatient', 'num_lab_procedures']

for ax, feat in zip(axes.flatten(), key_features):
    sns.boxplot(x='Readmission_30_Days', y=feat, data=data, hue='Readmission_30_Days',
                palette=['#3b82f6', '#ef4444'], ax=ax, legend=False)
    ax.set_title(f"{feat} vs Readmission_30_Days", fontweight="bold", fontsize=11)
    ax.set_xlabel("Readmission Class (0:No, 1:<30d)")
    ax.set_ylabel(feat)

plt.suptitle("Clinical Feature Distributions Stratified by 30-Day Readmission", fontsize=13, fontweight="bold")
plt.tight_layout()
boxplot_path = os.path.join(OUTPUT_FOLDER, "Boxplots.png")
plt.savefig(boxplot_path, dpi=300)
plt.close()

# ============================================================
# 8. SAVE RESULTS
# ============================================================
csv_path = os.path.join(OUTPUT_FOLDER, "correlation_matrix.csv")
corr_matrix.to_csv(csv_path)

print(f"\n[OK] Correlation matrix CSV saved to: {csv_path}")
print(f"[OK] Correlation heatmap saved to:   {heatmap_path}")
print(f"[OK] Numerical boxplots saved to:       {boxplot_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: CORRELATION & BOXPLOT ANALYSIS COMPLETED")
print("=" * 60)
