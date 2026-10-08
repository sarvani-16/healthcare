# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M1_Dataset_EDA"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

# ============================================================
# 4. LOAD DATASET
# ============================================================
print("=" * 60)
print("VITALSIGN - EXPLORATORY DATA ANALYSIS (EDA) (M1)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

# Create a clean working copy
data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (TARGET & MISSING VALUES)
# ============================================================
data = data.replace("?", np.nan)

# Create binary 30-day readmission target: <30 -> 1, >30 or NO -> 0
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

# ============================================================
# 6. MODEL / ANALYSIS & VISUALIZATION GENERATION
# ============================================================

# 1. 30-Day Readmission Distribution
print("\n[1] Generating 30-Day Readmission Distribution Plot...")
plt.figure(figsize=(7, 5))
counts = data['Readmission_30_Days'].value_counts()
ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#3b82f6", "#ef4444"], legend=False)
plt.title("30-Day Readmission Distribution", fontsize=13, fontweight="bold")
plt.xlabel("Readmission Class (0: No/<30d, 1: Readmitted <30d)", fontsize=11)
plt.ylabel("Number of Encounters", fontsize=11)
plt.xticks([0, 1], ["Not Readmitted (0)", "Readmitted <30d (1)"])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Readmission_Distribution.png"), dpi=300)
plt.close()

# 2. Age Distribution of Patients
print("[2] Generating Age Distribution of Patients...")
plt.figure(figsize=(10, 5))
order_age = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
sns.countplot(data=data, x='age', order=order_age, hue='age', palette='Blues_r', legend=False)
plt.title("Age Distribution of Patients", fontsize=13, fontweight="bold")
plt.xlabel("Age Bracket (Years)", fontsize=11)
plt.ylabel("Patient Count", fontsize=11)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Age_Distribution.png"), dpi=300)
plt.close()

# 3. Time in Hospital Distribution
print("[3] Generating Time in Hospital Distribution...")
plt.figure(figsize=(8, 5))
sns.histplot(data['time_in_hospital'], bins=14, kde=True, color="#2563eb")
plt.title("Time in Hospital Distribution", fontsize=13, fontweight="bold")
plt.xlabel("Length of Stay (Days)", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Time_in_Hospital_Distribution.png"), dpi=300)
plt.close()

# 4. Number of Medications
print("[4] Generating Number of Medications...")
plt.figure(figsize=(8, 5))
sns.histplot(data['num_medications'], bins=30, kde=True, color="#10b981")
plt.title("Number of Medications", fontsize=13, fontweight="bold")
plt.xlabel("Total Prescribed Medications", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Number_of_Medications.png"), dpi=300)
plt.close()

# 5. Number of Lab Procedures
print("[5] Generating Number of Lab Procedures...")
plt.figure(figsize=(8, 5))
sns.histplot(data['num_lab_procedures'], bins=30, kde=True, color="#8b5cf6")
plt.title("Number of Lab Procedures", fontsize=13, fontweight="bold")
plt.xlabel("Number of Lab Tests Administered", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Number_of_Lab_Procedures.png"), dpi=300)
plt.close()

# 6. Number of Diagnoses
print("[6] Generating Number of Diagnoses...")
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='number_diagnoses', hue='number_diagnoses', palette='Purples_r', legend=False)
plt.title("Number of Diagnoses", fontsize=13, fontweight="bold")
plt.xlabel("Number of Recorded Diagnoses", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Number_of_Diagnoses.png"), dpi=300)
plt.close()

# 7. Gender Distribution
print("[7] Generating Gender Distribution...")
plt.figure(figsize=(6, 5))
sns.countplot(data=data, x='gender', hue='gender', palette=['#3b82f6', '#ec4899', '#6b7280'], legend=False)
plt.title("Gender Distribution", fontsize=13, fontweight="bold")
plt.xlabel("Patient Gender", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Gender_Distribution.png"), dpi=300)
plt.close()

# 8. Race Distribution
print("[8] Generating Race Distribution...")
plt.figure(figsize=(9, 5))
race_counts = data['race'].fillna('Missing').value_counts()
sns.barplot(x=race_counts.index, y=race_counts.values, hue=race_counts.index, palette='crest', legend=False)
plt.title("Race Distribution", fontsize=13, fontweight="bold")
plt.xlabel("Race / Ethnicity", fontsize=11)
plt.ylabel("Patient Count", fontsize=11)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Race_Distribution.png"), dpi=300)
plt.close()

# 9. Admission Type Distribution
print("[9] Generating Admission Type Distribution...")
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='admission_type_id', hue='admission_type_id', palette='Spectral', legend=False)
plt.title("Admission Type Distribution", fontsize=13, fontweight="bold")
plt.xlabel("Admission Type ID (1:Emergency, 2:Urgent, 3:Elective)", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Admission_Type_Distribution.png"), dpi=300)
plt.close()

# 10. Boxplots
print("[10] Generating Numerical Boxplots...")
plt.figure(figsize=(12, 6))
num_box_cols = ['time_in_hospital', 'num_procedures', 'number_diagnoses', 'number_inpatient']
data[num_box_cols].plot(kind='box', subplots=True, layout=(1, 4), figsize=(12, 5), patch_artist=True)
plt.suptitle("Clinical Workload Feature Boxplots", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Boxplots.png"), dpi=300)
plt.close()

# 11. Correlation Heatmap
print("[11] Generating Correlation Heatmap...")
corr_features = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_inpatient', 'number_emergency',
    'number_diagnoses', 'Readmission_30_Days'
]
corr_matrix = data[corr_features].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Clinical Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Correlation_Heatmap.png"), dpi=300)
plt.close()

# 12. Missing-Value Visualization
print("[12] Generating Missing Values Visualization...")
missing_cnt = data.isnull().sum()[data.isnull().sum() > 0].sort_values(ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(x=missing_cnt.values, y=missing_cnt.index, hue=missing_cnt.index, palette='flare', legend=False)
plt.title("Missing Values per Clinical Attribute", fontsize=13, fontweight="bold")
plt.xlabel("Number of Missing Records", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Missing_Values_Visualization.png"), dpi=300)
plt.close()

# 13. Feature vs Target Plots
print("[13] Generating Feature vs Target Plots...")
plt.figure(figsize=(8, 5))
sns.boxplot(x='Readmission_30_Days', y='time_in_hospital', data=data, hue='Readmission_30_Days',
            palette=["#3b82f6", "#ef4444"], legend=False)
plt.title("Hospital Stay Duration vs 30-Day Readmission", fontsize=13, fontweight="bold")
plt.xlabel("Readmission Class", fontsize=11)
plt.ylabel("Time in Hospital (Days)", fontsize=11)
plt.xticks([0, 1], ["Not Readmitted (0)", "Readmitted <30d (1)"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Feature_vs_Target.png"), dpi=300)
plt.close()

# ============================================================
# 7. EVALUATION / SUMMARY
# ============================================================
readm_rate = (data['Readmission_30_Days'].mean()) * 100
print(f"\nReadmission Rate (<30 Days): {readm_rate:.2f}%")
print(f"Total Evaluated Features:    {len(corr_features)}")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
print(f"[OK] All 13 exploratory plots saved successfully to: {OUTPUT_FOLDER}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: EDA ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
