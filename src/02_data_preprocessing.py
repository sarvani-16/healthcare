"""
=============================================================================
VitalSign / HealthcarePrediction
Module 02: Data Preprocessing Pipeline
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Define Paths using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
MODELS_DIR = BASE_DIR / "models"
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

COLUMNS_TO_REMOVE = [
    'encounter_id',
    'patient_nbr',
    'weight',
    'payer_code',
    'medical_specialty',
    'readmitted'
]

def load_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    return df

def main():
    print("=" * 70)
    print("VITALSIGN: 02_DATA_PREPROCESSING")
    print("=" * 70)

    # 1. Load the dataset
    df = load_data(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Replace '?' with NaN
    df = df.replace("?", np.nan)
    print("Missing indicator '?' replaced with NaN.")

    # 3. Create Readmission_30_Days (<30 = 1, >30 or NO = 0)
    target_name = "Readmission_30_Days"
    df[target_name] = (
        df["readmitted"]
        .astype(str)
        .str.strip()
        .eq("<30")
        .astype(int)
    )
    print(f"Target column '{target_name}' created successfully:")
    print(f"  Class 1 (<30 Days Readmission): {df[target_name].sum()} ({df[target_name].mean()*100:.2f}%)")
    print(f"  Class 0 (No or >30 Days):       {(df[target_name] == 0).sum()} ({(1-df[target_name].mean())*100:.2f}%)")

    # 4. Remove columns: encounter_id, patient_nbr, weight, payer_code, medical_specialty, readmitted
    cols_to_drop = [c for c in COLUMNS_TO_REMOVE + [target_name] if c in df.columns]
    
    # 6. Separate X and y (Do not use target leakage)
    X = df.drop(columns=cols_to_drop)
    y = df[target_name]
    print(f"\nRemoved columns to prevent target leakage and uninformative features: {COLUMNS_TO_REMOVE}")
    print(f"Features matrix shape: {X.shape}, Target shape: {y.shape}")

    # 7. Automatically identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    print(f"\nIdentified {len(numerical_cols)} Numerical Columns:")
    print("  ", numerical_cols)
    print(f"Identified {len(categorical_cols)} Categorical Columns:")
    print("  ", categorical_cols)

    # Calculate baseline default values for form auto-completion
    default_values = {}
    for col in numerical_cols:
        default_values[col] = float(X[col].median(skipna=True))
    for col in categorical_cols:
        mode_s = X[col].mode(dropna=True)
        default_values[col] = str(mode_s[0]) if len(mode_s) > 0 else "Unknown"

    # 8. Numerical pipeline: SimpleImputer(strategy='median')
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 9. Categorical pipeline: SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 10. Use ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )

    print("\nFitting ColumnTransformer pipeline on feature matrix...")
    X_transformed = preprocessor.fit_transform(X)
    print(f"Feature transformation completed. Transformed feature matrix shape: {X_transformed.shape}")

    # Extract final feature names
    try:
        final_feature_columns = preprocessor.get_feature_names_out().tolist()
    except Exception:
        final_feature_columns = [f"Feature_{i}" for i in range(X_transformed.shape[1])]

    # 11. Save vitalsign_preprocessor.pkl and vitalsign_metadata.pkl
    preprocessor_path = MODELS_DIR / "vitalsign_preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n[OK] Saved Preprocessor to: {preprocessor_path}")

    metadata = {
        'original_feature_columns': X.columns.tolist(),
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'final_feature_columns': final_feature_columns,
        'target_name': target_name,
        'removed_columns': COLUMNS_TO_REMOVE,
        'default_values': default_values,
        'num_raw_features': X.shape[1],
        'num_encoded_features': X_transformed.shape[1],
        'classes': [0, 1]
    }
    metadata_path = MODELS_DIR / "vitalsign_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"[OK] Saved Metadata to:     {metadata_path}")

    print("\nData preprocessing completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
