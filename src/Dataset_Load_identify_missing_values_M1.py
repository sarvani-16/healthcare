"""
=============================================================================
VitalSign / HealthcarePrediction
Module 1: Dataset Loading & Missing Value Identification
File: Dataset_Load_identify_missing_values_M1.py
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "VitalSign_EDA_Analysis"
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

def load_dataset():
    """Loads the 50,000 healthcare dataset records and maps '?' to NaN."""
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

    df = df.replace("?", np.nan)
    return df

def main():
    print("=" * 70)
    print("VITALSIGN: DATASET LOAD & IDENTIFY MISSING VALUES (M1)")
    print("=" * 70)

    # 1. Load Dataset
    df = load_dataset()
    print(f"\n[1] DATASET LOADED: {df.shape[0]:,} Rows, {df.shape[1]} Columns")

    # First 5 rows
    print("\n--- First 5 Rows ---")
    print(df.head(5))

    # Display 6 columns subset
    print("\n--- Sample 6 Clinical Columns ---")
    print(df.iloc[:5, 2:8])

    # 2. Missing values per column
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    missing_pct = (missing_counts / len(df)) * 100

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing_counts.values,
        "Missing_Percentage": missing_pct.round(2).values
    })

    print("\n[2] MISSING VALUES ANALYSIS:")
    print(f"    Total Missing Values in Dataset: {total_missing:,}")
    print("\n--- Columns Containing Missing Values ---")
    cols_with_missing = missing_df[missing_df["Missing_Count"] > 0].sort_values(by="Missing_Count", ascending=False)
    print(cols_with_missing.to_string(index=False))

    # Save summary CSV
    summary_csv = OUTPUT_DIR / "Missing_Values_Summary.csv"
    missing_df.to_csv(summary_csv, index=False)
    print(f"\n[OK] Saved Missing Values Summary: {summary_csv}")

    # 3. Detect duplicate rows
    duplicate_rows = df[df.duplicated()]
    print(f"\n[3] DUPLICATE ROWS ANALYSIS:")
    print(f"    Total Duplicate Rows Detected: {len(duplicate_rows)}")

    # 4. Generate Missingness Heatmap
    print("\n[4] GENERATING MISSING VALUES HEATMAP...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("VitalSign Healthcare - Missing Values Heatmap (50,000 Records)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Dataset Attributes", fontweight="bold")
    plt.ylabel("Inpatient Records (50k)", fontweight="bold")

    heatmap_path = OUTPUT_DIR / "Missing_Values_Heatmap.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Missingness Heatmap: {heatmap_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
