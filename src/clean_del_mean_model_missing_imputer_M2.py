"""
=============================================================================
VitalSign / HealthcarePrediction
Module 2: Missing Value Handling via Deletion and Imputation
File: clean_del_mean_model_missing_imputer_M2.py
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Outputs: outputs/clean_imputed_demo_M2.csv
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

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
    print("VITALSIGN: MISSING VALUE DELETION & IMPUTATION (M2)")
    print("=" * 70)

    # 1. Read Dataset
    df = load_healthcare_data()
    print(f"Original Healthcare Dataset Loaded: {df.shape[0]:,} Rows, {df.shape[1]} Columns")

    # Show missing values before any treatment
    missing_before = df.isnull().sum()
    print("\n--- Top Missing Values Before Treatment ---")
    print(missing_before[missing_before > 0].sort_values(ascending=False).to_string())

    # ========================================================
    # TECHNIQUE A: DELETION DEMONSTRATION (Listwise Row Deletion)
    # ========================================================
    print("\n" + "-" * 60)
    print("TECHNIQUE A: LISTWISE ROW DELETION DEMO")
    print("-" * 60)
    df_deleted = df.dropna()
    print(f"Shape Before Deletion: {df.shape}")
    print(f"Shape After Deletion:  {df_deleted.shape}")
    print(f"Rows Dropped:          {len(df) - len(df_deleted):,} ({(len(df)-len(df_deleted))/len(df)*100:.1f}% loss)")
    print("Insight: Complete row deletion on medical data with sparsely recorded attributes")
    print("         (such as 'weight' with >96% missingness) causes severe data attrition.")
    print("         Therefore, targeted feature dropping + imputation is preferable.")

    # ========================================================
    # TECHNIQUE B & C: IMPUTATION (Mean/Median for Numeric, Mode for Categorical)
    # ========================================================
    print("\n" + "-" * 60)
    print("TECHNIQUE B & C: STATISTICAL IMPUTATION DEMO")
    print("-" * 60)
    df_imputed = df.copy()

    # Drop high-missing non-informative columns first
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df_imputed = df_imputed.drop(columns=[c for c in cols_to_drop if c in df_imputed.columns])

    num_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_imputed.select_dtypes(exclude=np.number).columns.tolist()

    # B. Numerical Median Imputation
    for col in num_cols:
        if df_imputed[col].isnull().sum() > 0:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)
            print(f"  [Numerical] Imputed {col} with Median = {median_val}")

    # C. Categorical Most-Frequent (Mode) Imputation
    for col in cat_cols:
        if df_imputed[col].isnull().sum() > 0:
            mode_val = df_imputed[col].mode(dropna=True)[0]
            df_imputed[col] = df_imputed[col].fillna(mode_val)
            print(f"  [Categorical] Imputed {col} with Mode = '{mode_val}'")

    missing_after = df_imputed.isnull().sum().sum()
    print(f"\nTotal Missing Values After Imputation: {missing_after}")
    print(f"Cleaned Imputed Shape:                 {df_imputed.shape}")

    # Save small processed demo
    out_csv = OUTPUT_DIR / "clean_imputed_demo_M2.csv"
    df_imputed.head(1000).to_csv(out_csv, index=False)
    print(f"\n[OK] Saved 1,000 imputed demo records to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
