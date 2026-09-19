"""
=============================================================================
VitalSign / HealthcarePrediction
Module: healthcare_dataset.py
Dataset understanding, cleaning, target generation, and summary reporting.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

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

def load_dataset(filepath=None) -> pd.DataFrame:
    """Safely loads the dataset from CSV, handling both header and headerless formats."""
    target_path = Path(filepath) if filepath else DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Healthcare dataset file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(target_path)
    else:
        df = pd.read_csv(target_path, header=None, names=COLUMN_NAMES)

    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces missing indicator '?' with standard NumPy NaN."""
    return df.replace("?", np.nan)

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the binary classification target: Readmission_30_Days
    <30 = 1 (Readmitted within 30 days)
    >30 or NO = 0 (Not readmitted within 30 days)
    """
    data = df.copy()
    if "readmitted" in data.columns and "Readmission_30_Days" not in data.columns:
        data["Readmission_30_Days"] = (
            data["readmitted"]
            .astype(str)
            .str.strip()
            .eq("<30")
            .astype(int)
        )
    return data

def get_dataset_shape(df: pd.DataFrame) -> tuple:
    """Returns dataset shape (rows, cols)."""
    return df.shape

def get_column_names(df: pd.DataFrame) -> list:
    """Returns list of column names."""
    return df.columns.tolist()

def get_data_types(df: pd.DataFrame) -> pd.Series:
    """Returns data types of all columns."""
    return df.dtypes

def get_missing_values(df: pd.DataFrame) -> pd.Series:
    """Returns missing value count per column after cleaning."""
    df_clean = clean_dataset(df)
    return df_clean.isnull().sum()

def get_duplicate_count(df: pd.DataFrame) -> int:
    """Returns count of duplicate records."""
    return int(df.duplicated().sum())

def get_target_distribution(df: pd.DataFrame) -> pd.Series:
    """Returns distribution of Readmission_30_Days."""
    data = create_target(df)
    return data["Readmission_30_Days"].value_counts(dropna=False)

def dataset_information(df: pd.DataFrame) -> dict:
    """Returns key structural dimensions and statistical summaries of the dataset."""
    df_clean = clean_dataset(df)
    data = create_target(df_clean)
    target_series = data.get("Readmission_30_Days")

    class_1 = int(target_series.sum()) if target_series is not None else 0
    total_len = len(df)
    readm_rate = (class_1 / total_len * 100) if total_len > 0 else 0.0

    return {
        "total_records": total_len,
        "total_features": df.shape[1],
        "duplicate_records": int(df.duplicated().sum()),
        "total_missing_cells": int(df_clean.isnull().sum().sum()),
        "target_name": "Readmission_30_Days",
        "class_1_count": class_1,
        "class_0_count": total_len - class_1,
        "readmission_rate_pct": round(readm_rate, 2),
        "columns": df.columns.tolist()
    }

def missing_value_report(df: pd.DataFrame, save_csv: bool = True) -> pd.DataFrame:
    """Generates column-wise missing value counts and percentages, saving to outputs/."""
    df_clean = clean_dataset(df)
    missing_series = df_clean.isnull().sum()
    missing_pct = (missing_series / len(df_clean)) * 100

    report_df = pd.DataFrame({
        "Column": df_clean.columns,
        "Missing_Count": missing_series.values,
        "Missing_Percentage": missing_pct.round(2).values
    })

    if save_csv:
        output_file = OUTPUTS_DIR / "missing_values.csv"
        report_df.to_csv(output_file, index=False)
        print(f"[OK] Missing values report saved to: {output_file}")

    return report_df

def duplicate_report(df: pd.DataFrame) -> int:
    """Calculates and reports number of duplicate encounter rows."""
    return get_duplicate_count(df)

def save_dataset_reports(df: pd.DataFrame):
    """Computes summary metrics and exports to dataset_summary.csv and missing_values.csv."""
    info = dataset_information(df)
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
            info["total_records"],
            info["total_features"],
            info["duplicate_records"],
            info["total_missing_cells"],
            info["target_name"],
            info["class_1_count"],
            info["class_0_count"],
            f"{info['readmission_rate_pct']}%"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    out_path = OUTPUTS_DIR / "dataset_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"[OK] Dataset summary saved to: {out_path}")

    missing_df = missing_value_report(df, save_csv=True)
    return summary_df, missing_df

def save_dataset_summary(df: pd.DataFrame):
    """Alias for backwards compatibility."""
    summary_df, _ = save_dataset_reports(df)
    return summary_df

if __name__ == "__main__":
    print("=" * 70)
    print("VITALSIGN: healthcare_dataset.py Execution")
    print("=" * 70)

    # 1. Load real CSV
    raw_df = load_dataset()
    print("Loaded real CSV successfully.")

    # 2. Replace '?' with NaN
    clean_df = clean_dataset(raw_df)
    print("Replaced '?' with NaN.")

    # 3. Print dataset shape
    shape = get_dataset_shape(clean_df)
    print(f"Dataset Shape: {shape[0]} rows, {shape[1]} columns")

    # 4. Print first five rows
    print("\n--- First 5 Rows ---")
    print(clean_df.head(5))

    # 5. Print column names
    print("\n--- Column Names ---")
    print(get_column_names(clean_df))

    # 6. Print data types
    print("\n--- Data Types ---")
    print(get_data_types(clean_df))

    # 7. Print missing values
    print("\n--- Missing Values (Non-zero) ---")
    missing = get_missing_values(clean_df)
    print(missing[missing > 0])

    # 8. Print duplicate count
    dup_count = get_duplicate_count(clean_df)
    print(f"\nDuplicate Records Count: {dup_count}")

    # 9. Print target distribution
    df_with_target = create_target(clean_df)
    print("\n--- Target Distribution (Readmission_30_Days) ---")
    target_dist = get_target_distribution(df_with_target)
    print(target_dist)

    # 10. Save dataset_summary.csv & missing_values.csv
    save_dataset_reports(df_with_target)

    print("\nDataset understanding completed successfully.")
    print("=" * 70)
