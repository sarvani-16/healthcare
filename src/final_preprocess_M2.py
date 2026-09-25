"""
=============================================================================
VITALSIGN: FINAL PREPROCESSING PIPELINE (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: Readmission_30_Days (<30 -> 1, >=30 or NO -> 0)
=============================================================================

ML Concept - Leakage-Free Preprocessing Pipeline:
1. Replaces '?' missing tokens with NaN.
2. Derives binary target: Readmission_30_Days.
3. Excludes high-cardinality leakage-prone IDs (encounter_id, patient_nbr)
   and ultra-sparse columns (>35% missing: weight, payer_code, medical_specialty).
4. Splits data FIRST into Train (80%) and Test (20%) using stratification.
5. Constructs scikit-learn ColumnTransformer:
   - Numerical: Median Imputation + StandardScaler
   - Categorical: Most Frequent Imputation + OneHotEncoder(sparse_output=False, handle_unknown='ignore')
6. Fits preprocessor strictly on X_train to prevent test data leakage.
7. Saves preprocessor artifact for Flask inference compatibility.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
    'weight', 'payer_code', 'medical_specialty'
]

def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)
    return df.replace("?", np.nan)

def build_preprocessing_pipeline(X: pd.DataFrame):
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

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

    return preprocessor, num_cols, cat_cols

def main():
    print("=" * 70)
    print("VITALSIGN: FINAL PREPROCESSING PIPELINE (M2)")
    print("=" * 70)

    # 1. Load dataset
    df = load_data()
    print(f"[1] Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")

    # 2. Create Target
    df['Readmission_30_Days'] = (df['readmitted'] == '<30').astype(int)
    print(f"[2] Target Created: 'Readmission_30_Days'")
    print(f"    Class 0 (No / >30d): {(df['Readmission_30_Days'] == 0).sum():,} ({(df['Readmission_30_Days'] == 0).mean():.2%})")
    print(f"    Class 1 (<30d Readmit): {(df['Readmission_30_Days'] == 1).sum():,} ({(df['Readmission_30_Days'] == 1).mean():.2%})")

    # 3. Separate Features and Target
    X = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns] + ['Readmission_30_Days'])
    y = df['Readmission_30_Days']
    print(f"\n[3] Predictor Features Matrix: {X.shape[1]} features (Dropped IDs & sparse columns)")

    # 4. Stratified Train/Test Split (80/20)
    print("\n[4] Performing Stratified 80/20 Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train Set: {X_train.shape[0]:,} records")
    print(f"    Test Set:  {X_test.shape[0]:,} records")

    # 5. Build and Fit Pipeline
    print("\n[5] Fitting ColumnTransformer on X_train (Zero Data Leakage)...")
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(X_train)
    preprocessor.fit(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"    Numerical Features Count:   {len(num_cols)}")
    print(f"    Categorical Features Count: {len(cat_cols)}")
    print(f"    Transformed Matrix Dimensions:")
    print(f"      - X_train_transformed: {X_train_transformed.shape}")
    print(f"      - X_test_transformed:  {X_test_transformed.shape}")

    # 6. Save Artifacts
    # Save processed demo CSV
    demo_sample = pd.DataFrame(
        X_test_transformed[:100],
        columns=[f"feat_{i}" for i in range(X_test_transformed.shape[1])]
    )
    demo_sample['target'] = y_test.iloc[:100].values
    demo_csv = OUTPUT_DIR / "final_preprocessed_sample_M2.csv"
    demo_sample.to_csv(demo_csv, index=False)
    print(f"\n[OK] Sample preprocessed dataset saved to: {demo_csv}")

    # Save preprocessor artifact for Flask app
    preprocessor_pkl = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_pkl)
    print(f"[OK] Preprocessor pipeline serialized to: {preprocessor_pkl}")

    print("\n" + "=" * 70)
    print("Preprocessing completed successfully. Zero models trained as requested.")
    print("=" * 70)

if __name__ == "__main__":
    main()