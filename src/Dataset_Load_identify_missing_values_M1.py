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
print("VITALSIGN - MISSING VALUE & INTEGRITY ANALYSIS (M1)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

# Create a copy so the original dataset is never modified
data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (REPLACE '?' WITH NaN)
# ============================================================
# In UCI Diabetes data, missing values are denoted by '?'
data = data.replace("?", np.nan)

# Duplicate records check
duplicate_count = int(data.duplicated().sum())

# ============================================================
# 6. MODEL / ANALYSIS
# ============================================================
missing_series = data.isnull().sum()
missing_pct_series = (missing_series / len(data)) * 100

missing_df = pd.DataFrame({
    'Column_Name': data.columns,
    'Missing_Count': missing_series.values,
    'Missing_Percentage': missing_pct_series.values.round(2)
})

# Filter columns that have missing values and sort descending
cols_with_missing = missing_df[missing_df['Missing_Count'] > 0].sort_values(
    by='Missing_Count', ascending=False
)

total_missing = int(missing_series.sum())
print(f"Total Missing Values across all cells: {total_missing:,}")
print(f"Total Duplicate Records: {duplicate_count}")
print("\n--- COLUMNS WITH MISSING VALUES ---")
print(cols_with_missing.to_string(index=False))

# ============================================================
# 7. EVALUATION / VISUALIZATION
# ============================================================
# Generate Missing Values Heatmap (sample of 1,000 encounters for clear rendering)
plt.figure(figsize=(12, 6))
sample_missing = data.sample(n=min(1000, len(data)), random_state=42).isnull()
sns.heatmap(sample_missing, cbar=True, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap (Sampled 1,000 Encounters)", fontsize=14, fontweight="bold")
plt.xlabel("Dataset Attributes", fontsize=11)
plt.ylabel("Hospital Encounters", fontsize=11)
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_FOLDER, "Missing_Values_Heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

# ============================================================
# 8. SAVE RESULTS
# ============================================================
summary_csv_path = os.path.join(OUTPUT_FOLDER, "Missing_Values_Summary.csv")
cols_with_missing.to_csv(summary_csv_path, index=False)

print(f"\n[OK] Missing values summary CSV saved to: {summary_csv_path}")
print(f"[OK] Missing values heatmap plot saved to: {heatmap_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: MISSING VALUES ANALYSIS COMPLETED")
print("=" * 60)
