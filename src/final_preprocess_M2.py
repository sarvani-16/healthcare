# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M2_Preprocessing"
MODELS_DIR = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/models"
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
print("VITALSIGN - FINAL PREPROCESSING PIPELINE (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (CLEANING & SPLITTING)
# ============================================================
# Replace '?' with NaN
data = data.replace("?", np.nan)

# Create Readmission_30_Days (<30 -> 1, >30 or NO -> 0)
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

# Separate X and y
X = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns] + ['Readmission_30_Days'])
y = data['Readmission_30_Days']

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Total Predictor Features: {X.shape[1]} ({len(num_cols)} numerical, {len(cat_cols)} categorical)")

# Stratified 80/20 Train/Test Split (Preventing Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Set Encounters: {len(X_train):,}")
print(f"Testing Set Encounters:  {len(X_test):,}")

# ============================================================
# 6. MODEL / ANALYSIS (BUILD COLUMNTRANSFORMER PIPELINE)
# ============================================================
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
)

# Fit strictly on training set only (prevent leakage)
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# ============================================================
# 7. EVALUATION / SUMMARY
# ============================================================
print(f"\nTransformed X_train Shape: {X_train_transformed.shape}")
print(f"Transformed X_test Shape:  {X_test_transformed.shape}")

summary_df = pd.DataFrame({
    'Step': [
        '1. Raw Dataset Shape',
        '2. Target Readmission_30_Days Rate',
        '3. Excluded Unsuitable Columns',
        '4. Numerical Features (Median + StandardScaled)',
        '5. Categorical Features (Mode + OneHotEncoded)',
        '6. Train Split Encounters (80%)',
        '7. Test Split Encounters (20%)',
        '8. Transformed Features Dimension'
    ],
    'Details': [
        str(df.shape),
        f"{y.mean():.2%}",
        str(DROP_COLUMNS),
        str(len(num_cols)),
        str(len(cat_cols)),
        f"{len(X_train):,}",
        f"{len(X_test):,}",
        str(X_train_transformed.shape[1])
    ]
})

print("\n--- PREPROCESSING PIPELINE SUMMARY ---")
print(summary_df.to_string(index=False))

# ============================================================
# 8. SAVE RESULTS
# ============================================================
out_csv = os.path.join(OUTPUT_FOLDER, "final_preprocessed_summary.csv")
summary_df.to_csv(out_csv, index=False)

# Save fitted preprocessor for Flask deployment
prep_pkl = os.path.join(MODELS_DIR, "preprocessor.pkl")
joblib.dump(preprocessor, prep_pkl)

print(f"\n[OK] Preprocessing summary saved to: {out_csv}")
print(f"[OK] Fitted preprocessor saved to:   {prep_pkl}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: FINAL PREPROCESSING PIPELINE READY")
print("=" * 60)