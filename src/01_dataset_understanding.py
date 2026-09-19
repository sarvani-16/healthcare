"""
=============================================================================
VitalSign / HealthcarePrediction
Module 01: Dataset Understanding & Exploration
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Load the dataset using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard UCI Diabetes 130-US Hospitals column names (50 columns)
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

def load_data(filepath: Path) -> pd.DataFrame:
    # 2. Check whether the dataset exists
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
    print("VITALSIGN: 01_DATASET_UNDERSTANDING")
    print("=" * 70)

    # 1 & 2. Load dataset with pathlib and check existence
    df = load_data(DATASET_PATH)
    print("Dataset loaded successfully.")

    # 4. Display dataset shape
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Display first five rows
    print("\n--- First Five Rows ---")
    print(df.head())

    # Display column names
    print("\n--- Column Names ---")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx:2d}. {col}")

    # Display data types
    print("\n--- Data Types ---")
    print(df.dtypes)

    # 3. Replace '?' with NaN
    df_clean = df.replace("?", np.nan)
    print("\nReplaced missing indicator '?' with NaN.")

    # Missing values
    missing_series = df_clean.isnull().sum()
    missing_pct = (missing_series / len(df_clean)) * 100
    missing_df = pd.DataFrame({
        "Column": df_clean.columns,
        "Missing_Count": missing_series.values,
        "Missing_Percentage": missing_pct.round(2).values
    })
    print("\n--- Missing Values Summary ---")
    print(missing_df[missing_df["Missing_Count"] > 0].sort_values(by="Missing_Count", ascending=False).to_string(index=False))

    # Duplicate count
    dup_count = int(df.duplicated().sum())
    print(f"\nDuplicate Records Count: {dup_count}")

    # Unique values for important columns
    important_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'insulin', 'change', 'diabetesMed', 'readmitted']
    print("\n--- Unique Values for Important Columns ---")
    for col in important_cols:
        if col in df.columns:
            uniques = df[col].dropna().unique()
            print(f"  {col} ({len(uniques)} unique values): {list(uniques[:10])}")

    # 5. Create Readmission_30_Days (<30 = 1, >30 or NO = 0)
    if "readmitted" in df.columns:
        df["Readmission_30_Days"] = (
            df["readmitted"]
            .astype(str)
            .str.strip()
            .eq("<30")
            .astype(int)
        )

        # 6. Display target distribution
        print("\n--- Original Target ('readmitted') Distribution ---")
        print(df["readmitted"].value_counts(dropna=False))

        print("\n--- Formulated Target ('Readmission_30_Days') Distribution ---")
        print(df["Readmission_30_Days"].value_counts())
        class_1_count = int(df["Readmission_30_Days"].sum())
        class_0_count = int((df["Readmission_30_Days"] == 0).sum())
        rate_30 = (df["Readmission_30_Days"].mean() * 100)
        print(f"Class 1 (<30 Days Readmitted): {class_1_count} ({rate_30:.2f}%)")
        print(f"Class 0 (No / >30 Days): {class_0_count} ({100 - rate_30:.2f}%)")

    # 7. Save outputs
    summary_data = {
        "Metric": [
            "Total Records",
            "Total Features",
            "Duplicate Records",
            "Total Missing Values (with ? as NaN)",
            "Target Column",
            "Class 1 (<30 Days Readmitted)",
            "Class 0 (No or >30 Days)",
            "Readmission Rate (%)"
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            dup_count,
            int(df_clean.isnull().sum().sum()),
            "readmitted -> Readmission_30_Days",
            class_1_count if "readmitted" in df.columns else "N/A",
            class_0_count if "readmitted" in df.columns else "N/A",
            f"{rate_30:.2f}%" if "readmitted" in df.columns else "N/A"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_file = OUTPUTS_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n[OK] Saved dataset summary to: {summary_file}")

    missing_file = OUTPUTS_DIR / "missing_values.csv"
    missing_df.to_csv(missing_file, index=False)
    print(f"Missing values saved.")

    print("\nDataset understanding completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()
