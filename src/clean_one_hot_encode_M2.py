"""
=============================================================================
VitalSign / HealthcarePrediction
Module 2: Categorical Feature Encoding - One-Hot Encoding
File: clean_one_hot_encode_M2.py (and clean_one_hot_encod_M2.py)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Outputs: outputs/clean_one_hot_encode_demo_M2.csv
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
    print("VITALSIGN: ONE-HOT ENCODING DEMONSTRATION (M2)")
    print("=" * 70)

    # 1. Load Dataset
    df = load_healthcare_data()
    print(f"Dataset Loaded: {df.shape[0]:,} Rows, {df.shape[1]} Columns")

    # Select representative categorical clinical features
    selected_cats = ['race', 'gender', 'max_glu_serum', 'insulin', 'diabetesMed']
    print(f"\n[1] Selected Categorical Columns for One-Hot Encoding:")
    print(f"    {selected_cats}")

    # Prepare demo subset with missing values filled
    demo_df = df[selected_cats].head(10).copy()
    for col in selected_cats:
        demo_df[col] = demo_df[col].fillna("Missing")

    print("\n[2] Original Categorical Records (First 10 Rows):")
    print(demo_df)

    # 2. Fit OneHotEncoder
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_array = ohe.fit_transform(demo_df)
    feature_names = ohe.get_feature_names_out(selected_cats)

    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=demo_df.index)

    print(f"\n[3] Encoded Columns Generated ({len(feature_names)} Binary Indicators):")
    for feat in feature_names:
        print(f"    - {feat}")

    print(f"\n[4] Shapes Comparison:")
    print(f"    Original Shape: {demo_df.shape} (5 categorical columns)")
    print(f"    Encoded Shape:  {encoded_df.shape} ({encoded_df.shape[1]} binary indicator columns)")

    print("\n--- Encoded Binary Matrix Sample (First 5 Rows) ---")
    print(encoded_df.iloc[:5, :8])

    out_csv = OUTPUT_DIR / "clean_one_hot_encode_demo_M2.csv"
    encoded_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved One-Hot demonstration CSV to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
