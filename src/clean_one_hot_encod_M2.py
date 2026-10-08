# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
print("VITALSIGN - ONE-HOT ENCODING DEMONSTRATION (M2)")
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
selected_cats = ['race', 'gender', 'max_glu_serum', 'insulin', 'diabetesMed']
demo_df = data[selected_cats].fillna('Missing').copy()

print("\nOriginal Selected Categorical Columns:")
print(selected_cats)
print(f"Original Shape: {demo_df.shape}")

# ============================================================
# 6. MODEL / ANALYSIS (ONE-HOT ENCODER)
# ============================================================
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_arr = ohe.fit_transform(demo_df)
encoded_feature_names = ohe.get_feature_names_out(selected_cats)

encoded_df = pd.DataFrame(encoded_arr, columns=encoded_feature_names)

# ============================================================
# 7. EVALUATION / COMPARISON
# ============================================================
print(f"\nEncoded Feature Matrix Shape: {encoded_df.shape}")
print(f"Generated Binary Feature Columns ({len(encoded_feature_names)}):")
for f_name in encoded_feature_names[:10]:
    print(f"  - {f_name}")
if len(encoded_feature_names) > 10:
    print(f"  ... and {len(encoded_feature_names) - 10} more indicator columns.")

sample_combined = pd.concat([demo_df.head(15).reset_index(drop=True),
                             encoded_df.head(15).reset_index(drop=True)], axis=1)

# ============================================================
# 8. SAVE RESULTS
# ============================================================
out_csv = os.path.join(OUTPUT_FOLDER, "one_hot_encoded_sample.csv")
sample_combined.to_csv(out_csv, index=False)
print(f"\n[OK] One-hot encoded sample CSV saved to: {out_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: ONE-HOT ENCODING DEMONSTRATION COMPLETED")
print("=" * 60)
