# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
print("VITALSIGN - TARGET MEAN ENCODING (LEAKAGE-FREE) (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (CREATE Readmission_30_Days)
# ============================================================
data = data.replace("?", np.nan)
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

feature_col = 'race'
data[feature_col] = data[feature_col].fillna('Missing_Category')

# CRITICAL LEAKAGE PREVENTION:
# Split FIRST into Train (80%) and Test (20%) using stratification.
X_train, X_test, y_train, y_test = train_test_split(
    data[[feature_col, 'encounter_id']],
    data['Readmission_30_Days'],
    test_size=0.20,
    random_state=42,
    stratify=data['Readmission_30_Days']
)

print(f"Train Encounters: {len(X_train):,} | Test Encounters: {len(X_test):,}")

# ============================================================
# 6. MODEL / ANALYSIS (FIT TARGET ENCODING ON TRAIN ONLY)
# ============================================================
global_train_mean = y_train.mean()

train_target_stats = pd.DataFrame({
    'category': X_train[feature_col],
    'target': y_train
}).groupby('category')['target'].agg(['count', 'mean']).rename(
    columns={'count': 'Train_Count', 'mean': 'Target_Mean'}
)

print("\n--- LEARNED TARGET ENCODING STATISTICS (TRAIN SPLIT ONLY) ---")
print(train_target_stats.to_string())

# Map learned training means to Train and Test sets
target_map = train_target_stats['Target_Mean'].to_dict()
X_test_encoded = X_test[feature_col].map(target_map).fillna(global_train_mean)

# ============================================================
# 7. EVALUATION / SUMMARY
# ============================================================
demo_output = pd.DataFrame({
    'encounter_id': X_test['encounter_id'].head(20),
    'race_category': X_test[feature_col].head(20),
    'target_encoded_value': X_test_encoded.head(20).round(4),
    'actual_readmission_label': y_test.head(20).values
})

print("\n--- SAMPLE TARGET ENCODED TEST SAMPLES (LEAKAGE-FREE) ---")
print(demo_output.to_string(index=False))

# ============================================================
# 8. SAVE RESULTS
# ============================================================
out_csv = os.path.join(OUTPUT_FOLDER, "target_encoded.csv")
demo_output.to_csv(out_csv, index=False)
print(f"\n[OK] Target encoded demonstration saved to: {out_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: TARGET ENCODING DEMONSTRATION COMPLETED")
print("=" * 60)
