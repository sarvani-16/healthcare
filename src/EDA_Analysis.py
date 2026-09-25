"""
=============================================================================
VitalSign / HealthcarePrediction
Module 1: Exploratory Data Analysis (EDA)
File: EDA_Analysis.py
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: Readmission_30_Days (<30 -> 1, >30 or NO -> 0)
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

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "font.size": 10})

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

def load_and_prepare_data():
    """Loads dataset, handles '?' as NaN, and derives binary Readmission_30_Days."""
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

    df = df.replace("?", np.nan)

    # Create binary target Readmission_30_Days
    # <30 = 1 (readmitted within 30 days)
    # >30 / NO = 0 (no readmission within 30 days)
    df["Readmission_30_Days"] = (
        df["readmitted"].astype(str).str.strip().eq("<30").astype(int)
    )
    return df

def detect_outliers_iqr(df, num_cols):
    """Detects outliers using the standard Interquartile Range (IQR) method."""
    print("\n--- IQR-BASED OUTLIER DETECTION ---")
    outlier_summary = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary.append({
            "Feature": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower_Bound": round(lower_bound, 2),
            "Upper_Bound": round(upper_bound, 2),
            "Outlier_Count": len(outliers),
            "Outlier_Percentage": round(len(outliers) / len(df) * 100, 2)
        })
        print(f"  {col:<22}: {len(outliers):5d} outliers ({len(outliers)/len(df)*100:5.2f}%) [Limits: {lower_bound:.1f} to {upper_bound:.1f}]")
    return pd.DataFrame(outlier_summary)

def main():
    print("=" * 70)
    print("VITALSIGN: COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    df = load_and_prepare_data()
    print(f"Dataset Loaded: {df.shape[0]:,} Rows, {df.shape[1]} Columns")

    # 1. TARGET ANALYSIS: Readmission_30_Days
    print("\n[1] TARGET DISTRIBUTION (Readmission_30_Days):")
    target_counts = df["Readmission_30_Days"].value_counts()
    for val, count in target_counts.items():
        label = "Readmitted <30 Days (Class 1)" if val == 1 else "Not Readmitted / >30 Days (Class 0)"
        print(f"    {label}: {count:,} ({count/len(df)*100:.2f}%)")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Class 0 (>30d / NO)", "Class 1 (<30d Readmit)"], target_counts.values, color=["#2563eb", "#dc2626"], width=0.55)
    ax.set_title("VitalSign: 30-Day Readmission Target Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Patient Encounters", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, weight='bold')
    plt.savefig(OUTPUT_DIR / "Target_Distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Target Distribution: {OUTPUT_DIR / 'Target_Distribution.png'}")

    # 2. MISSING VALUES HEATMAP
    print("\n[2] GENERATING MISSING VALUES HEATMAP...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("VitalSign Healthcare - Missing Values Heatmap", fontsize=12, fontweight="bold", pad=12)
    plt.savefig(OUTPUT_DIR / "Missing_Values_Heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Missingness Heatmap: {OUTPUT_DIR / 'Missing_Values_Heatmap.png'}")

    # 3. UNIVARIATE ANALYSIS: Categorical Features (Age Distribution)
    print("\n[3] UNIVARIATE ANALYSIS (Age Distribution)...")
    fig, ax = plt.subplots(figsize=(8, 5))
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_counts = df['age'].value_counts().reindex(age_order).fillna(0)
    ax.bar(age_counts.index, age_counts.values, color="#0284c7")
    ax.set_title("Patient Age Bracket Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Age Bracket", fontweight="bold")
    ax.set_ylabel("Number of Encounters", fontweight="bold")
    plt.xticks(rotation=30)
    plt.savefig(OUTPUT_DIR / "Age_Distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Age Distribution: {OUTPUT_DIR / 'Age_Distribution.png'}")

    # 4. UNIVARIATE ANALYSIS: Numerical Feature (Time in Hospital Boxplot)
    print("\n[4] TIME IN HOSPITAL BOXPLOT...")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x=df['time_in_hospital'], color="#38bdf8", ax=ax)
    ax.set_title("Distribution of Time in Hospital (Days)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Length of Stay (Days)", fontweight="bold")
    plt.savefig(OUTPUT_DIR / "Time_in_Hospital_Boxplot.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Time in Hospital Boxplot: {OUTPUT_DIR / 'Time_in_Hospital_Boxplot.png'}")

    # 5. BIVARIATE ANALYSIS: Feature vs Target (Inpatient Visits by Readmission Status)
    print("\n[5] BIVARIATE ANALYSIS (Feature vs Target)...")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Readmission_30_Days', y='time_in_hospital', data=df, palette=["#3b82f6", "#ef4444"], ax=ax)
    ax.set_title("Length of Stay vs 30-Day Readmission Status", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticklabels(["No Readmit / >30d (0)", "Readmitted <30d (1)"])
    ax.set_xlabel("Clinical Readmission Outcome", fontweight="bold")
    ax.set_ylabel("Time in Hospital (Days)", fontweight="bold")
    plt.savefig(OUTPUT_DIR / "Feature_vs_Target.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Feature vs Target: {OUTPUT_DIR / 'Feature_vs_Target.png'}")

    # 6. CORRELATION MATRIX & HEATMAP
    print("\n[6] CORRELATION ANALYSIS...")
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'Readmission_30_Days']
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap of Key Clinical Metrics", fontsize=12, fontweight="bold", pad=12)
    plt.savefig(OUTPUT_DIR / "Correlation_Heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Correlation Heatmap: {OUTPUT_DIR / 'Correlation_Heatmap.png'}")

    # 7. OUTLIER ANALYSIS (IQR)
    outlier_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_inpatient']
    detect_outliers_iqr(df, outlier_cols)

    # 8. PAIR PLOT (Sampled subset for high efficiency)
    print("\n[7] GENERATING SAMPLED PAIRPLOT (1,000 Sampled Encounters)...")
    sample_df = df[['time_in_hospital', 'num_medications', 'num_lab_procedures', 'Readmission_30_Days']].dropna().sample(n=1000, random_state=42)
    pairplot_fig = sns.pairplot(sample_df, hue='Readmission_30_Days', palette={0: "#2563eb", 1: "#dc2626"}, markers=["o", "s"], plot_kws={"alpha": 0.6})
    pairplot_fig.fig.suptitle("Pairplot of Clinical Features by Readmission Status (Sample n=1,000)", y=1.02, fontsize=12, fontweight="bold")
    pairplot_fig.savefig(OUTPUT_DIR / "Pairplot.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Sampled Pairplot: {OUTPUT_DIR / 'Pairplot.png'}")

    print("\n" + "=" * 70)
    print("EDA ANALYSIS COMPLETED SUCCESSFULLY. ALL GRAPHS SAVED.")
    print("=" * 70)

if __name__ == "__main__":
    main()
