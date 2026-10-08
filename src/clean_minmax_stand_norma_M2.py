# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M2_Preprocessing"
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
print("VITALSIGN - FEATURE SCALING (MINMAX & STANDARD SCALER) (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (SELECT NUMERICAL ATTRIBUTES)
# ============================================================
num_cols = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses'
]

raw_df = data[num_cols].dropna().copy()

# ============================================================
# 6. MODEL / ANALYSIS (APPLY MINMAX & STANDARD SCALER)
# ============================================================
# 1. MinMaxScaler [0, 1]
minmax = MinMaxScaler()
minmax_arr = minmax.fit_transform(raw_df)
minmax_df = pd.DataFrame(minmax_arr, columns=[f"{col}_minmax" for col in num_cols])

# 2. StandardScaler (mean=0, std=1)
standard = StandardScaler()
standard_arr = standard.fit_transform(raw_df)
standard_df = pd.DataFrame(standard_arr, columns=[f"{col}_std" for col in num_cols])

# ============================================================
# 7. EVALUATION / SUMMARY STATISTICS
# ============================================================
print("\n--- RAW FEATURES SUMMARY (BEFORE SCALING) ---")
print(raw_df.describe().transpose()[['mean', 'std', 'min', 'max']])

print("\n--- MINMAX SCALED SUMMARY (RANGE [0, 1]) ---")
print(minmax_df.describe().transpose()[['mean', 'std', 'min', 'max']].round(3))

print("\n--- STANDARD SCALED SUMMARY (MEAN ~ 0, STD = 1) ---")
print(standard_df.describe().transpose()[['mean', 'std', 'min', 'max']].round(3))

combined_sample = pd.concat([
    raw_df.head(10).add_suffix('_raw'),
    minmax_df.head(10),
    standard_df.head(10)
], axis=1)

# ============================================================
# 8. SAVE RESULTS
# ============================================================
out_csv = os.path.join(OUTPUT_FOLDER, "scaled_features.csv")
combined_sample.to_csv(out_csv, index=False)
print(f"\n[OK] Scaled features comparison CSV saved to: {out_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: FEATURE SCALING COMPLETED")
print("=" * 60)
