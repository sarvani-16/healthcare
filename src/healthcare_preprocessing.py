"""
=============================================================================
VitalSign / HealthcarePrediction
Module: healthcare_preprocessing.py
Replaces and enhances M2 preprocessing, encoding, and scaling pipelines.
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

try:
    from .healthcare_dataset import load_dataset, clean_dataset, create_target
except ImportError:
    from healthcare_dataset import load_dataset, clean_dataset, create_target

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS_TO_REMOVE = [
    'encounter_id',
    'patient_nbr',
    'readmitted',
    'weight',
    'payer_code',
    'medical_specialty'
]

def prepare_features_target(df: pd.DataFrame, target_col: str = "Readmission_30_Days"):
    """
    Separates predictors X and target y, removing leakage and direct identifier columns.
    Excludes: encounter_id, patient_nbr, readmitted, weight, payer_code, medical_specialty.
    """
    data = clean_dataset(df)
    data = create_target(data)

    cols_to_drop = [c for c in COLUMNS_TO_REMOVE + [target_col] if c in data.columns]
    X = data.drop(columns=cols_to_drop)
    y = data[target_col]

    return X, y

def build_preprocessing_pipeline(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Constructs scikit-learn ColumnTransformer:
    - Numerical: SimpleImputer(strategy='median')
    - Categorical: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
    """
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, numerical_cols),
            ('cat', cat_pipe, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor

def preprocess_dataset(X: pd.DataFrame, preprocessor: ColumnTransformer = None, fit: bool = False):
    """
    Transforms feature matrix X using ColumnTransformer.
    If fit=True, fits and returns (X_transformed, fitted_preprocessor).
    """
    if fit:
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
        X_trans = preprocessor.fit_transform(X)
        return X_trans, preprocessor
    else:
        if preprocessor is None:
            preprocessor = load_preprocessor()
        X_trans = preprocessor.transform(X)
        return X_trans

def save_preprocessor(preprocessor: ColumnTransformer, filename: str = "preprocessor.pkl"):
    """Saves fitted ColumnTransformer to models/ directory."""
    out_path = MODELS_DIR / filename
    joblib.dump(preprocessor, out_path)
    # Also save with vitalsign_ prefix for full cross-compatibility
    alias_path = MODELS_DIR / "vitalsign_preprocessor.pkl"
    joblib.dump(preprocessor, alias_path)
    print(f"[OK] Saved Preprocessor to: {out_path} and {alias_path}")
    return out_path

def load_preprocessor(filename: str = "preprocessor.pkl") -> ColumnTransformer:
    """Loads fitted ColumnTransformer from models/ directory."""
    path = MODELS_DIR / filename
    if not path.exists():
        alias = MODELS_DIR / "vitalsign_preprocessor.pkl"
        if alias.exists():
            path = alias
        else:
            raise FileNotFoundError(f"Preprocessor not found at {path}. Run training first.")
    return joblib.load(path)

if __name__ == "__main__":
    print("=" * 70)
    print("VITALSIGN: healthcare_preprocessing.py Execution")
    print("=" * 70)
    data = load_dataset()
    X, y = prepare_features_target(data)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    X_trans, preprocessor = preprocess_dataset(X, fit=True)
    print(f"Transformed matrix shape: {X_trans.shape}")

    save_preprocessor(preprocessor)
    print("Preprocessing completed successfully.")
    print("=" * 70)
