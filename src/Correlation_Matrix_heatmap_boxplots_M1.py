"""
=============================================================================
VitalSign / HealthcarePrediction
Module 1: Correlation Matrix, Heatmap & Boxplots Analysis
File: Correlation_Matrix_heatmap_boxplots_M1.py
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: Readmission_30_Days (<30 -> 1, >=30 / NO -> 0)
Outputs: outputs/Boxplots_Correlation/
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "Boxplots_Correlation"
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
    """Loads dataset and formulates binary Readmission_30_Days target."""
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

    df = df.replace("?", np.nan)
    df["Readmission_30_Days"] = df["readmitted"].astype(str).str.strip().eq("<30").astype(int)
    return df

def main():
    print("=" * 70)
    print("VITALSIGN: CORRELATION MATRIX & BOXPLOTS ANALYSIS (M1)")
    print("=" * 70)

    # 1. Load dataset
    df = load_healthcare_data()
    print(f"Dataset Loaded Successfully: {df.shape[0]:,} Rows, {df.shape[1]} Columns")

    # 2. Select numerical columns (excluding patient identifiers)
    excluded_cols = ['encounter_id', 'patient_nbr']
    numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded_cols]
    print(f"\nNumerical Columns Selected for Correlation ({len(numerical_cols)}):")
    print(numerical_cols)

    # 3. Compute Correlation Matrix
    corr_matrix = df[numerical_cols].corr()
    print("\n--- Correlation Matrix ---")
    print(corr_matrix.round(3))

    # Save Correlation Matrix CSV
    corr_csv_path = OUTPUT_DIR / "Correlation_Matrix.csv"
    corr_matrix.to_csv(corr_csv_path)
    print(f"\n[OK] Saved Correlation Matrix CSV: {corr_csv_path}")

    # 4. Generate Correlation Heatmap
    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("VitalSign Healthcare - Numerical Features Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    heatmap_path = OUTPUT_DIR / "Correlation_Heatmap.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Correlation Heatmap: {heatmap_path}")

    # 5. Produce Boxplots for Numerical Features vs Readmission_30_Days
    print("\nGenerating Boxplots: Numerical Features vs Readmission_30_Days...")
    target_col = "Readmission_30_Days"
    features_to_plot = [c for c in numerical_cols if c != target_col]

    for col in features_to_plot:
        plt.figure(figsize=(6, 5))
        sns.boxplot(
            x=target_col,
            y=col,
            data=df,
            palette=["#3b82f6", "#ef4444"],
            hue=target_col,
            legend=False
        )
        plt.title(f"{col} vs Readmission_30_Days", fontsize=12, fontweight="bold", pad=10)
        plt.xlabel("Readmitted <30 Days (0 = No/Late, 1 = Yes)", fontweight="bold")
        plt.ylabel(col, fontweight="bold")
        plt.xticks([0, 1], ["No / >30d (0)", "Readmitted <30d (1)"])

        boxplot_path = OUTPUT_DIR / f"Boxplot_{col}_vs_{target_col}.png"
        plt.savefig(boxplot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Saved boxplot: {boxplot_path.name}")

    print("\n" + "=" * 70)
    print(f"All correlation & boxplot tasks completed. Outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
