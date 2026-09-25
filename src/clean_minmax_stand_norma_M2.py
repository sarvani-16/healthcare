"""
=============================================================================
VITALSIGN: FEATURE SCALING - MINMAX & STANDARD SCALER (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Features: time_in_hospital, num_lab_procedures, num_procedures,
          num_medications, number_diagnoses
=============================================================================

ML Concept - Feature Scaling:
Distance-based and gradient-based algorithms (such as Logistic Regression,
KNN, Neural Networks, SVM) are sensitive to feature magnitude. Features with
larger raw values dominate objective functions.

1. MinMaxScaler:
   Transforms features to a bounded range [0, 1]:
   X_scaled = (X - X_min) / (X_max - X_min)
   Useful when bounded ranges are required and extreme outliers are minimal.

2. StandardScaler (Z-Score Normalization):
   Centers features around zero with unit variance:
   Z = (X - mean) / std
   Standard choice for linear models and regularization penalties.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)
    return df.replace("?", np.nan)

def main():
    print("=" * 70)
    print("VITALSIGN: FEATURE SCALING (MINMAX & STANDARD SCALER) (M2)")
    print("=" * 70)

    df = load_data()
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")

    scaling_features = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_diagnoses'
    ]

    raw_subset = df[scaling_features].copy()

    # [1] Raw Statistics
    print("\n[1] RAW NUMERICAL FEATURES STATISTICS (Before Scaling):")
    raw_stats = raw_subset.describe().T[['mean', 'std', 'min', 'max']]
    print(raw_stats.to_string())

    # [2] MinMaxScaler
    print("\n[2] APPLYING MINMAX SCALER (Bounded Range: [0.0, 1.0]):")
    minmax = MinMaxScaler()
    minmax_scaled = pd.DataFrame(
        minmax.fit_transform(raw_subset),
        columns=[f"{col}_minmax" for col in scaling_features]
    )
    minmax_stats = minmax_scaled.describe().T[['mean', 'std', 'min', 'max']]
    print(minmax_stats.round(4).to_string())

    # [3] StandardScaler
    print("\n[3] APPLYING STANDARD SCALER (Zero Mean, Unit Variance):")
    standard = StandardScaler()
    standard_scaled = pd.DataFrame(
        standard.fit_transform(raw_subset),
        columns=[f"{col}_standard" for col in scaling_features]
    )
    standard_stats = standard_scaled.describe().T[['mean', 'std', 'min', 'max']]
    print(standard_stats.round(4).to_string())

    # [4] Side-by-side demonstration for first 5 patients
    demo_sample = pd.concat([
        raw_subset.head(5).add_suffix('_raw'),
        minmax_scaled.head(5),
        standard_scaled.head(5)
    ], axis=1)

    print("\n[4] Side-by-side comparison for 'time_in_hospital' (First 5 Patients):")
    print(demo_sample[['time_in_hospital_raw', 'time_in_hospital_minmax', 'time_in_hospital_standard']].to_string())

    # Save demo CSV
    out_csv = OUTPUT_DIR / "clean_scaling_demo_M2.csv"
    demo_sample.to_csv(out_csv, index=False)
    print(f"\n[OK] Feature scaling comparison saved to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
