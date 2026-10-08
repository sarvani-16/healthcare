# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator

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
print("VITALSIGN - MISSING VALUE TREATMENT COMPARISON (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (REPLACE '?' WITH NaN)
# ============================================================
data = data.replace("?", np.nan)
initial_missing = data.isnull().sum()
print(f"Dataset Shape: {data.shape}")
print(f"Total Missing Cells Before Cleaning: {initial_missing.sum():,}")

# Select a representative numerical column and categorical column
num_col = 'num_lab_procedures'
cat_col = 'race'

# Introduce small artificial NaNs in num_col for demonstration if none exist
demo_data = data[[num_col, cat_col, 'medical_specialty', 'time_in_hospital']].copy()
demo_data.loc[demo_data.sample(frac=0.05, random_state=42).index, num_col] = np.nan

# ============================================================
# 6. MODEL / ANALYSIS (DEMONSTRATE 6 IMPUTATION TECHNIQUES)
# ============================================================
results = []

# Method 1: Listwise Row Deletion
data_del = demo_data.dropna()
results.append({
    'Method': 'Listwise Row Deletion',
    'Feature': 'All Demo Features',
    'Original_Missing': int(demo_data.isnull().sum().sum()),
    'Remaining_Rows': len(data_del),
    'Data_Loss_Percent': round((1 - len(data_del) / len(demo_data)) * 100, 2),
    'Strategy_Type': 'Deletion'
})

# Method 2: Mean Imputation (Numerical)
mean_imputer = SimpleImputer(strategy='mean')
mean_imputed = mean_imputer.fit_transform(demo_data[[num_col]])
results.append({
    'Method': 'Mean Imputation',
    'Feature': num_col,
    'Original_Missing': int(demo_data[num_col].isnull().sum()),
    'Remaining_Rows': len(demo_data),
    'Data_Loss_Percent': 0.0,
    'Strategy_Type': 'Statistical Imputation'
})

# Method 3: Median Imputation (Numerical - Robust to Outliers)
median_imputer = SimpleImputer(strategy='median')
median_imputed = median_imputer.fit_transform(demo_data[[num_col]])
results.append({
    'Method': 'Median Imputation',
    'Feature': num_col,
    'Original_Missing': int(demo_data[num_col].isnull().sum()),
    'Remaining_Rows': len(demo_data),
    'Data_Loss_Percent': 0.0,
    'Strategy_Type': 'Statistical Imputation'
})

# Method 4: Most Frequent / Mode Imputation (Categorical)
mode_imputer = SimpleImputer(strategy='most_frequent')
mode_imputed = mode_imputer.fit_transform(demo_data[[cat_col]])
results.append({
    'Method': 'Most Frequent (Mode)',
    'Feature': cat_col,
    'Original_Missing': int(demo_data[cat_col].isnull().sum()),
    'Remaining_Rows': len(demo_data),
    'Data_Loss_Percent': 0.0,
    'Strategy_Type': 'Categorical Imputation'
})

# Method 5: Constant Imputation ('Missing_Not_Recorded')
const_imputer = SimpleImputer(strategy='constant', fill_value='Missing_Not_Recorded')
const_imputed = const_imputer.fit_transform(demo_data[['medical_specialty']])
results.append({
    'Method': 'Constant Flag Imputation',
    'Feature': 'medical_specialty',
    'Original_Missing': int(demo_data['medical_specialty'].isnull().sum()),
    'Remaining_Rows': len(demo_data),
    'Data_Loss_Percent': 0.0,
    'Strategy_Type': 'Domain Heuristic'
})

# Method 6: Missing Indicator Feature Flag
indicator = MissingIndicator()
missing_flags = indicator.fit_transform(demo_data[[num_col]])
results.append({
    'Method': 'Missing Indicator Flag',
    'Feature': f"{num_col}_was_missing",
    'Original_Missing': int(demo_data[num_col].isnull().sum()),
    'Remaining_Rows': len(demo_data),
    'Data_Loss_Percent': 0.0,
    'Strategy_Type': 'Binary Indicator'
})

# ============================================================
# 7. EVALUATION / SUMMARY
# ============================================================
comparison_df = pd.DataFrame(results)
print("\n--- MISSING VALUE TREATMENT COMPARISON TABLE ---")
print(comparison_df.to_string(index=False))

# ============================================================
# 8. SAVE RESULTS
# ============================================================
csv_out_path = os.path.join(OUTPUT_FOLDER, "missing_value_methods.csv")
comparison_df.to_csv(csv_out_path, index=False)
print(f"\n[OK] Comparison results saved to: {csv_out_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: MISSING VALUE IMPUTATION DEMO COMPLETED")
print("=" * 60)
