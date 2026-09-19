"""
=============================================================================
VitalSign / HealthcarePrediction
Module: healthcare_eda.py
Replaces and enhances M1 exploratory data analysis and visualization.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .healthcare_dataset import load_dataset, clean_dataset, create_target
except ImportError:
    from healthcare_dataset import load_dataset, clean_dataset, create_target

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_CHARTS_DIR = BASE_DIR / "static" / "charts"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.autolayout": True
})

def _save_plot(fig, filename: str):
    """Internal helper to save plot to outputs/ and static/charts/."""
    out_path = OUTPUTS_DIR / filename
    static_path = STATIC_CHARTS_DIR / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(static_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {filename}")
    return str(out_path)

def create_readmission_chart(df: pd.DataFrame, filename: str = "readmission_distribution.png"):
    """Generates and saves the target Readmission Distribution chart."""
    if "readmitted" not in df.columns:
        print("[WARN] 'readmitted' column not found in DataFrame.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['readmitted'].value_counts()
    colors = ['#2563eb', '#0ea5e9', '#ef4444']
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.55)
    ax.set_title("Target Readmission Distribution (UCI Diabetes)", pad=15, fontweight="bold")
    ax.set_xlabel("Readmission Status", fontweight="bold")
    ax.set_ylabel("Patient Encounters", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold')
    return _save_plot(fig, filename)

def create_age_chart(df: pd.DataFrame, filename: str = "age_distribution.png"):
    """Generates and saves the Patient Age Distribution chart."""
    if "age" not in df.columns:
        print("[WARN] 'age' column not found in DataFrame.")
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    age_order = [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ]
    age_counts = df['age'].value_counts().reindex(age_order).fillna(0)
    ax.bar(age_counts.index, age_counts.values, color='#0284c7')
    ax.set_title("Patient Age Bracket Distribution", pad=15, fontweight="bold")
    ax.set_xlabel("Age Group", fontweight="bold")
    ax.set_ylabel("Number of Patients", fontweight="bold")
    plt.xticks(rotation=30)
    return _save_plot(fig, filename)

def create_gender_chart(df: pd.DataFrame, filename: str = "gender_distribution.png"):
    """Generates and saves the Patient Gender Breakdown pie chart."""
    if "gender" not in df.columns:
        print("[WARN] 'gender' column not found in DataFrame.")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    gender_clean = df['gender'].replace('Unknown/Invalid', np.nan).dropna()
    gender_counts = gender_clean.value_counts()
    ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
           startangle=140, colors=['#3b82f6', '#ec4899'], explode=(0.02, 0.02),
           wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
    ax.set_title("Patient Gender Breakdown", pad=15, fontweight="bold")
    return _save_plot(fig, filename)

def create_race_chart(df: pd.DataFrame, filename: str = "race_distribution.png"):
    """Generates and saves the Patient Race / Ethnicity horizontal bar chart."""
    if "race" not in df.columns:
        print("[WARN] 'race' column not found in DataFrame.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    race_counts = df['race'].fillna('Missing').value_counts()
    ax.barh(race_counts.index[::-1], race_counts.values[::-1], color='#0d9488')
    ax.set_title("Patient Race / Ethnicity Distribution", pad=15, fontweight="bold")
    ax.set_xlabel("Patient Encounters", fontweight="bold")
    return _save_plot(fig, filename)

def create_admission_type_chart(df: pd.DataFrame, filename: str = "admission_type_distribution.png"):
    """Generates and saves the Admission Type Breakdown chart."""
    if "admission_type_id" not in df.columns:
        print("[WARN] 'admission_type_id' column not found in DataFrame.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    adm_labels = {
        1: 'Emergency', 2: 'Urgent', 3: 'Elective',
        4: 'Newborn', 5: 'Not Available', 6: 'NULL', 7: 'Trauma Center'
    }
    adm_counts = df['admission_type_id'].value_counts().head(6)
    x_labels = [f"{k} ({adm_labels.get(k, 'Other')})" for k in adm_counts.index]
    ax.bar(x_labels, adm_counts.values, color='#6366f1')
    ax.set_title("Admission Type Distribution", pad=15, fontweight="bold")
    ax.set_xlabel("Admission Type ID & Category", fontweight="bold")
    ax.set_ylabel("Encounters Count", fontweight="bold")
    plt.xticks(rotation=25, ha='right')
    return _save_plot(fig, filename)

def create_correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap.png"):
    """Generates and saves the numerical correlation heatmap."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    clean_num_cols = [c for c in numeric_cols if c not in ['encounter_id', 'patient_nbr']]

    if len(clean_num_cols) < 2:
        print("[WARN] Insufficient numerical columns for correlation heatmap.")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[clean_num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True,
                cbar_kws={"shrink": 0.8}, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap of Clinical Numerical Features", pad=15, fontweight="bold")
    return _save_plot(fig, filename)

def save_all_charts(df: pd.DataFrame):
    """Executes all EDA chart functions and saves results."""
    print("Generating and saving all EDA visualization charts...")
    create_readmission_chart(df)
    create_age_chart(df)
    create_gender_chart(df)
    create_race_chart(df)
    create_admission_type_chart(df)
    create_correlation_heatmap(df)
    print("All clinical EDA charts saved successfully.")

if __name__ == "__main__":
    print("=" * 70)
    print("VITALSIGN: healthcare_eda.py Execution")
    print("=" * 70)
    data = load_dataset()
    data = clean_dataset(data)
    save_all_charts(data)
    print("=" * 70)
