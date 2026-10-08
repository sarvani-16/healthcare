# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M1_Dataset_EDA"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Standard UCI Diabetes 130-US Hospitals column names (50 features)
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
print("VITALSIGN - DATASET UNDERSTANDING & EXPLORATION (M1)")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Healthcare dataset not found at: {DATASET_PATH}")

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

# Create a processing copy
data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (UNDERSTANDING INSPECTION)
# ============================================================
num_rows, num_cols = data.shape
num_features = num_cols - 1  # Excluding target readmitted

num_cols_list = data.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_list = data.select_dtypes(exclude=[np.number]).columns.tolist()

# ============================================================
# 6. MODEL / ANALYSIS
# ============================================================
print("\n[DATASET PROPERTIES]")
print(f"Dataset Shape:        {data.shape}")
print(f"Number of Rows:       {num_rows:,}")
print(f"Number of Columns:    {num_cols}")
print(f"Number of Features:   {num_features}")

print(f"\nNumerical Features ({len(num_cols_list)}):")
print(num_cols_list)

print(f"\nCategorical Features ({len(cat_cols_list)}):")
print(cat_cols_list)

print("\nTarget Column: 'readmitted'")
print(data['readmitted'].value_counts(dropna=False))

print("\n--- FIRST 10 ROWS ---")
print(data.head(10))

print("\n--- LAST 10 ROWS ---")
print(data.tail(10))

print("\n--- DATA TYPES ---")
print(data.dtypes)

print("\n--- DATAFRAME INFO ---")
data.info()

print("\n--- STATISTICAL SUMMARY (NUMERICAL) ---")
stat_summary = data.describe().transpose()
print(stat_summary)

# ============================================================
# 7. EVALUATION / SUMMARY METRICS
# ============================================================
summary_df = pd.DataFrame({
    'Metric': [
        'Total Encounters (Rows)',
        'Total Features (Columns)',
        'Numerical Features Count',
        'Categorical Features Count',
        'Target Column',
        'Duplicate Encounters'
    ],
    'Value': [
        str(num_rows),
        str(num_cols),
        str(len(num_cols_list)),
        str(len(cat_cols_list)),
        'readmitted',
        str(data.duplicated().sum())
    ]
})

# ============================================================
# 8. SAVE RESULTS
# ============================================================
summary_path = os.path.join(OUTPUT_FOLDER, "dataset_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n[OK] Dataset summary saved to: {summary_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: DATASET UNDERSTANDING COMPLETED SUCCESSFULLY")
print("=" * 60)
