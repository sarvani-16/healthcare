# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M4_Clustering"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
print("VITALSIGN - PRINCIPAL COMPONENT ANALYSIS (PCA) (M4)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (CONTINUOUS NUMERICAL FEATURES & SCALING)
# ============================================================
data = data.replace("?", np.nan)
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_diagnoses',
    'number_inpatient',
    'number_emergency',
    'number_outpatient'
]

print(f"Selected Clinical Numerical Features ({len(features)}):")
print(f"  {features}")

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(data[features])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ============================================================
# 6. MODEL / ANALYSIS (PCA FIT & EIGENVECTOR DECOMPOSITION)
# ============================================================
pca_full = PCA()
pca_full.fit(X_scaled)

exp_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

print("\n--- PCA EXPLAINED VARIANCE RATIO ---")
for i, (ev, cv) in enumerate(zip(exp_var, cum_var), 1):
    print(f"  PC{i}: {ev:.4f} ({ev:.2%}) | Cumulative: {cv:.4f} ({cv:.2%})")

# Fit 2-Component PCA for Projection
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Loadings matrix
loadings_df = pd.DataFrame(
    pca_2d.components_.T,
    index=features,
    columns=['PC1_Loading', 'PC2_Loading']
)
print("\n--- PRINCIPAL COMPONENT LOADINGS ---")
print(loadings_df.round(4).to_string())

# ============================================================
# 7. EVALUATION / SUMMARY
# ============================================================
summary_df = pd.DataFrame({
    'Principal_Component': [f'PC{i}' for i in range(1, len(features) + 1)],
    'Explained_Variance_Ratio': [round(x, 4) for x in exp_var],
    'Cumulative_Variance': [round(x, 4) for x in cum_var]
})

# ============================================================
# 8. SAVE RESULTS (PLOTS & CSV)
# ============================================================
# 1. Explained Variance & Scree Plot
fig, ax1 = plt.subplots(figsize=(8, 5))
x_axis = range(1, len(features) + 1)
ax1.bar(x_axis, exp_var, alpha=0.6, color='#3b82f6', label='Individual Variance')
ax1.set_xlabel('Principal Component', fontsize=10)
ax1.set_ylabel('Individual Explained Variance', color='#3b82f6', fontsize=10)
ax1.set_xticks(list(x_axis))

ax2 = ax1.twinx()
ax2.plot(x_axis, cum_var, color='#ef4444', marker='o', linewidth=2, label='Cumulative Variance')
ax2.set_ylabel('Cumulative Explained Variance', color='#ef4444', fontsize=10)
ax2.set_ylim(0, 1.05)

plt.title('PCA Explained Variance Scree Plot', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

var_png = os.path.join(OUTPUT_FOLDER, "PCA_Explained_Variance.png")
plt.savefig(var_png, dpi=300)
plt.close()

# 2. 2D PCA Scatter Plot (Sample 3,000 encounters for clear display)
plt.figure(figsize=(9, 6))
sample_indices = np.random.RandomState(42).choice(len(X_pca_2d), size=3000, replace=False)
scatter = plt.scatter(
    X_pca_2d[sample_indices, 0],
    X_pca_2d[sample_indices, 1],
    c=data['Readmission_30_Days'].iloc[sample_indices],
    cmap='coolwarm',
    alpha=0.6,
    s=25,
    edgecolors='none'
)
plt.colorbar(scatter, label="30-Day Readmission (0 = No, 1 = Yes)")
plt.title("PCA 2D Clinical Patient Feature Space Projection", fontsize=12, fontweight="bold")
plt.xlabel(f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

pca2d_png = os.path.join(OUTPUT_FOLDER, "PCA_2D.png")
plt.savefig(pca2d_png, dpi=300)
plt.close()

# 3. Save Loadings and Summary CSVs
loadings_csv = os.path.join(OUTPUT_FOLDER, "pca_loadings.csv")
loadings_df.to_csv(loadings_csv)

summary_csv = os.path.join(OUTPUT_FOLDER, "pca_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print(f"\n[OK] Scree plot saved to:          {var_png}")
print(f"[OK] 2D PCA plot saved to:         {pca2d_png}")
print(f"[OK] PCA loadings saved to:        {loadings_csv}")
print(f"[OK] PCA summary saved to:         {summary_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: PRINCIPAL COMPONENT ANALYSIS COMPLETED")
print("=" * 60)
