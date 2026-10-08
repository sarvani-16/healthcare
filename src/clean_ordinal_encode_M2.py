# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

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
print("VITALSIGN - ORDINAL ENCODING DEMONSTRATION (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (ORDERED ATTRIBUTE SELECTION)
# ============================================================
# ACADEMIC NOTE ON ORDINAL ENCODING:
# Ordinal encoding is strictly appropriate ONLY when categories have an inherent,
# natural mathematical ranking (e.g., age groups, tumor grades, education tiers).
# It must NEVER be applied to purely nominal attributes (such as 'race', 'gender',
# or 'admission_type_id') because assigning arbitrary integers injects false
# Euclidean distance and mathematical magnitude into linear and distance algorithms.

age_categories_ordered = [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
]

demo_df = data[['encounter_id', 'age']].copy()

# ============================================================
# 6. MODEL / ANALYSIS (ORDINAL ENCODER)
# ============================================================
ordinal_encoder = OrdinalEncoder(
    categories=[age_categories_ordered],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

encoded_age = ordinal_encoder.fit_transform(demo_df[['age']])
demo_df['age_ordinal_encoded'] = encoded_age.astype(int)

# ============================================================
# 7. EVALUATION / SUMMARY
# ============================================================
print(f"Selected Ordinal Clinical Feature: 'age'")
print(f"Explicit Chronological Hierarchy: {age_categories_ordered}")
print("\n--- SAMPLE ENCODED AGE COMPARISON (FIRST 15 PATIENTS) ---")
print(demo_df.head(15).to_string(index=False))

# ============================================================
# 8. SAVE RESULTS
# ============================================================
out_csv = os.path.join(OUTPUT_FOLDER, "ordinal_encoded.csv")
demo_df.head(100).to_csv(out_csv, index=False)
print(f"\n[OK] Ordinal encoded sample CSV saved to: {out_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: ORDINAL ENCODING DEMONSTRATION COMPLETED")
print("=" * 60)
