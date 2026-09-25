"""
=============================================================================
VitalSign / HealthcarePrediction
Module 1: Dataset Loading and Understanding
File: load_understand_dataset.py
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import io

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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

def load_healthcare_data(filepath=DATASET_PATH):
    """Safely loads diabetic dataset, handling header or headerless format."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    # Clean missing indicator '?' with NaN
    df = df.replace("?", np.nan)
    return df

def main():
    print("=" * 70)
    print("VITALSIGN: DATASET LOADING & UNDERSTANDING")
    print("=" * 70)

    df = load_healthcare_data()

    # 1. Dataset Shape
    print(f"\n[1] DATASET SHAPE:")
    print(f"    Rows:    {df.shape[0]:,}")
    print(f"    Columns: {df.shape[1]}")

    # 2. First 5 records
    print(f"\n[2] FIRST 5 RECORDS:")
    print(df.head(5))

    # 3. Last 5 records
    print(f"\n[3] LAST 5 RECORDS:")
    print(df.tail(5))

    # 4. Column Names
    print(f"\n[4] COLUMN NAMES ({len(df.columns)} Total):")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

    # 5. Data Types
    print(f"\n[5] DATA TYPES:")
    print(df.dtypes)

    # 6. df.info()
    print(f"\n[6] DATAFRAME SUMMARY INFO:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)

    # 7. Numerical & Categorical Columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    print(f"\n[7] NUMERICAL COLUMNS ({len(num_cols)}):")
    print(f"    {num_cols}")
    print(f"\n[8] CATEGORICAL COLUMNS ({len(cat_cols)}):")
    print(f"    {cat_cols}")

    # 9. Missing Values
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]
    print(f"\n[9] MISSING VALUES PER COLUMN (Total Missing Cells: {missing.sum():,}):")
    if not missing_nonzero.empty:
        for col, cnt in missing_nonzero.items():
            pct = (cnt / len(df)) * 100
            print(f"    {col:<24}: {cnt:6d} ({pct:5.2f}%)")
    else:
        print("    No missing values detected.")

    # 10. Duplicate Records
    dup_count = int(df.duplicated().sum())
    print(f"\n[10] DUPLICATE RECORDS:")
    print(f"    Total Duplicates: {dup_count}")

    # 11. Statistical Summary
    print(f"\n[11] STATISTICAL SUMMARY (Numerical Features):")
    print(df.describe().T)

    # Save summary report
    summary_txt = OUTPUTS_DIR / "dataset_understanding_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("VITALSIGN HEALTHCARE DATASET UNDERSTANDING SUMMARY\n")
        f.write(f"Total Rows: {df.shape[0]:,}\n")
        f.write(f"Total Columns: {df.shape[1]}\n")
        f.write(f"Duplicate Count: {dup_count}\n")
        f.write(f"Total Missing Values: {missing.sum():,}\n\n")
        f.write("Columns:\n" + ", ".join(df.columns) + "\n\n")
        f.write("Data Info:\n" + info_str + "\n\n")
        f.write("Statistical Summary:\n" + df.describe().T.to_string() + "\n")
    print(f"\n[OK] Summary saved to: {summary_txt}")
    print("=" * 70)

if __name__ == "__main__":
    main()
