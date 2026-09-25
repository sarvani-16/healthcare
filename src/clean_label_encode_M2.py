"""
=============================================================================
VitalSign / HealthcarePrediction
Module 2: Categorical Feature Encoding - Label Encoding
File: clean_label_encode_M2.py
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Outputs: outputs/clean_label_encode_demo_M2.csv
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def load_healthcare_data():
    """Loads dataset and replaces '?' missing symbol with NaN."""
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

    return df.replace("?", np.nan)

def main():
    print("=" * 70)
    print("VITALSIGN: CATEGORICAL LABEL ENCODING (M2)")
    print("=" * 70)

    # 1. Load Dataset
    df = load_healthcare_data()
    print(f"Dataset Loaded: {df.shape[0]:,} Rows, {df.shape[1]} Columns")

    # =========================================================================
    # ML CONCEPT EXPLANATION: WHEN TO USE LABEL ENCODING
    # =========================================================================
    # In clinical machine learning:
    # 1. LabelEncoder maps categories to integers: [Category_A -> 0, Category_B -> 1, ...]
    # 2. Linear and distance-based models (like Logistic Regression or SVM) interpret
    #    these integers as numerical magnitudes (e.g., 2 > 1 > 0).
    # 3. For nominal clinical categories without intrinsic rank (such as 'race' or
    #    'medical_specialty'), Label Encoding falsely imposes a hierarchy.
    #    Hence, Label Encoding should be reserved for binary categories (gender,
    #    diabetesMed, change) or target variables, whereas One-Hot Encoding
    #    is mathematically sound for nominal predictors.
    # =========================================================================

    selected_cols = ['gender', 'change', 'diabetesMed']
    print(f"\nSelected Categorical Columns for Demonstration: {selected_cols}")

    demo_df = df[selected_cols].copy().dropna().head(10)
    encoded_df = pd.DataFrame()

    encoders = {}
    for col in selected_cols:
        le = LabelEncoder()
        encoded_col = le.fit_transform(demo_df[col].astype(str))
        encoders[col] = le
        encoded_df[col + "_original"] = demo_df[col].values
        encoded_df[col + "_encoded"] = encoded_col

        print(f"\n--- Label Encoding Mapping for: {col} ---")
        for cls_name, cls_code in zip(le.classes_, range(len(le.classes_))):
            print(f"    '{cls_name}' -> {cls_code}")

    print("\n--- Comparison of Original vs Encoded Values (First 10 Encounters) ---")
    print(encoded_df)

    out_csv = OUTPUT_DIR / "clean_label_encode_demo_M2.csv"
    encoded_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Demonstration saved to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
