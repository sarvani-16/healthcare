# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
print("VITALSIGN - LABEL ENCODING DEMONSTRATION (M2)")
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
# Select appropriate binary/ordinal categorical features
# CAUTION: Do NOT blindly label encode nominal features like race or admission_type_id
selected_cols = ['gender', 'change', 'diabetesMed']
demo_df = data[selected_cols].dropna().copy()

# ============================================================
# 6. MODEL / ANALYSIS (LABEL ENCODING)
# ============================================================
encoded_records = {}

for col in selected_cols:
    le = LabelEncoder()
    encoded_vals = le.fit_transform(demo_df[col].astype(str))
    encoded_records[f"{col}_original"] = demo_df[col].values
    encoded_records[f"{col}_encoded"] = encoded_vals
    
    print(f"\nFeature: '{col}'")
    for cls_idx, cls_label in enumerate(le.classes_):
        print(f"  Mapping: '{cls_label}' -> {cls_idx}")

# ============================================================
# 7. EVALUATION / SUMMARY
# ============================================================
comparison_df = pd.DataFrame(encoded_records).head(20)
print("\n--- SAMPLE LABEL ENCODED VALUES (FIRST 20 PATIENTS) ---")
print(comparison_df.to_string())

# ============================================================
# 8. SAVE RESULTS
# ============================================================
out_csv = os.path.join(OUTPUT_FOLDER, "label_encoded.csv")
comparison_df.to_csv(out_csv, index=False)
print(f"\n[OK] Label encoded results saved to: {out_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: LABEL ENCODING DEMONSTRATION COMPLETED")
print("=" * 60)
