"""
=============================================================================
VitalSign / HealthcarePrediction
Module 03: Exploratory Data Analysis (EDA)
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Use matplotlib with the Agg backend (headless server/CLI compatibility)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_CHARTS_DIR = BASE_DIR / "static" / "charts"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

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

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.autolayout": True
})

def load_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    df = df.replace("?", np.nan)
    return df

def save_chart(fig, filename: str):
    out_file = OUTPUTS_DIR / filename
    static_file = STATIC_CHARTS_DIR / filename
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    fig.savefig(static_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved chart: {filename}")

def main():
    print("=" * 70)
    print("VITALSIGN: 03_EDA_ANALYSIS")
    print("=" * 70)

    df = load_data(DATASET_PATH)
    print(f"Dataset loaded for EDA. Shape: {df.shape}")

    # 1. Readmission distribution
    if "readmitted" in df.columns:
        print("Generating Readmission Distribution Chart...")
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = df['readmitted'].value_counts()
        colors = ['#2563eb', '#0ea5e9', '#ef4444']
        bars = ax.bar(counts.index, counts.values, color=colors, width=0.55)
        ax.set_title("Target Readmission Distribution (UCI Diabetes)", pad=15, fontweight="bold")
        ax.set_xlabel("Readmission Category", fontweight="bold")
        ax.set_ylabel("Patient Encounter Count", fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, weight='bold')
        save_chart(fig, "readmission_distribution.png")
    else:
        print("[WARNING] Column 'readmitted' unavailable for Readmission Distribution Chart.")

    # 2. Age distribution
    if "age" in df.columns:
        print("Generating Age Distribution Chart...")
        fig, ax = plt.subplots(figsize=(9, 5))
        age_order = [
            '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
            '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
        ]
        age_counts = df['age'].value_counts().reindex(age_order).fillna(0)
        ax.bar(age_counts.index, age_counts.values, color='#0284c7')
        ax.set_title("Patient Age Bracket Distribution", pad=15, fontweight="bold")
        ax.set_xlabel("Age Bracket", fontweight="bold")
        ax.set_ylabel("Number of Patients", fontweight="bold")
        plt.xticks(rotation=30)
        save_chart(fig, "age_distribution.png")
    else:
        print("[WARNING] Column 'age' unavailable for Age Distribution Chart.")

    # 3. Gender distribution
    if "gender" in df.columns:
        print("Generating Gender Distribution Chart...")
        fig, ax = plt.subplots(figsize=(7, 5))
        gender_clean = df['gender'].replace('Unknown/Invalid', np.nan).dropna()
        gender_counts = gender_clean.value_counts()
        ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
               startangle=140, colors=['#3b82f6', '#ec4899'], explode=(0.02, 0.02),
               wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
        ax.set_title("Patient Gender Breakdown", pad=15, fontweight="bold")
        save_chart(fig, "gender_distribution.png")
    else:
        print("[WARNING] Column 'gender' unavailable for Gender Distribution Chart.")

    # 4. Race distribution
    if "race" in df.columns:
        print("Generating Race Distribution Chart...")
        fig, ax = plt.subplots(figsize=(8, 5))
        race_counts = df['race'].fillna('Missing').value_counts()
        ax.barh(race_counts.index[::-1], race_counts.values[::-1], color='#0d9488')
        ax.set_title("Patient Race / Ethnicity Distribution", pad=15, fontweight="bold")
        ax.set_xlabel("Patient Encounters", fontweight="bold")
        save_chart(fig, "race_distribution.png")
    else:
        print("[WARNING] Column 'race' unavailable for Race Distribution Chart.")

    # 5. Admission type distribution
    if "admission_type_id" in df.columns:
        print("Generating Admission Type Distribution Chart...")
        fig, ax = plt.subplots(figsize=(8, 5))
        adm_labels = {
            1: 'Emergency', 2: 'Urgent', 3: 'Elective',
            4: 'Newborn', 5: 'Not Available', 6: 'NULL', 7: 'Trauma Center'
        }
        adm_counts = df['admission_type_id'].value_counts().head(6)
        x_labels = [f"{k} ({adm_labels.get(k, 'Other')})" for k in adm_counts.index]
        ax.bar(x_labels, adm_counts.values, color='#6366f1')
        ax.set_title("Admission Type Distribution", pad=15, fontweight="bold")
        ax.set_xlabel("Admission Type & Clinical Category", fontweight="bold")
        ax.set_ylabel("Encounters Count", fontweight="bold")
        plt.xticks(rotation=25, ha='right')
        save_chart(fig, "admission_type_distribution.png")
    else:
        print("[WARNING] Column 'admission_type_id' unavailable for Admission Type Chart.")

    # 6. Discharge disposition distribution
    if "discharge_disposition_id" in df.columns:
        print("Generating Discharge Disposition Distribution Chart...")
        fig, ax = plt.subplots(figsize=(9, 5))
        disch_labels = {
            1: 'Discharged Home', 2: 'Short-term Hospital', 3: 'SNF',
            4: 'ICF', 5: 'Inpatient Care', 6: 'Home Health', 7: 'AMA',
            18: 'Hospice', 22: 'Rehab'
        }
        disch_counts = df['discharge_disposition_id'].value_counts().head(7)
        x_disch = [f"ID {k}: {disch_labels.get(k, 'Other')}" for k in disch_counts.index]
        ax.bar(x_disch, disch_counts.values, color='#059669')
        ax.set_title("Discharge Disposition Categories", pad=15, fontweight="bold")
        ax.set_xlabel("Discharge Destination", fontweight="bold")
        ax.set_ylabel("Encounters Count", fontweight="bold")
        plt.xticks(rotation=30, ha='right')
        save_chart(fig, "discharge_disposition_distribution.png")
    else:
        print("[WARNING] Column 'discharge_disposition_id' unavailable for Discharge Disposition Chart.")

    # 7. Correlation heatmap for numerical columns
    print("Generating Correlation Heatmap...")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    clean_num_cols = [c for c in numeric_cols if c not in ['encounter_id', 'patient_nbr']]

    if len(clean_num_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[clean_num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True,
                    cbar_kws={"shrink": 0.8}, ax=ax, linewidths=0.5)
        ax.set_title("Correlation Heatmap of Clinical Numerical Features", pad=15, fontweight="bold")
        save_chart(fig, "correlation_heatmap.png")
    else:
        print("[WARNING] Insufficient numerical columns for Correlation Heatmap.")

    print("\nEDA analysis completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
