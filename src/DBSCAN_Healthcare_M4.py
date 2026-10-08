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

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M4_Clustering"
MODELS_DIR = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/models"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
print("VITALSIGN - DBSCAN DENSITY-BASED CLUSTERING (M4)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (SAMPLING & STANDARDIZATION)
# ============================================================
data = data.replace("?", np.nan)

features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_diagnoses'
]

# Use 5,000 representative records for density-based spatial clustering
sample_df = data[features].dropna().sample(n=5000, random_state=42)

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(sample_df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ============================================================
# 6. MODEL / ANALYSIS (DBSCAN CLUSTERING)
# ============================================================
# eps=1.2, min_samples=10 for clinical density discovery
dbscan = DBSCAN(eps=1.2, min_samples=10)
labels = dbscan.fit_predict(X_scaled)

sample_df['Cluster'] = labels
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nDBSCAN Clustering Results:")
print(f"  - Number of Clusters Formed: {n_clusters}")
print(f"  - Number of Noise/Outliers:  {n_noise} ({n_noise / len(labels):.2%})")

# ============================================================
# 7. EVALUATION / CLUSTER SUMMARY
# ============================================================
cluster_counts = pd.Series(labels).value_counts().sort_index()
print("\n--- DBSCAN CLUSTER DISTRIBUTION ---")
for cl, count in cluster_counts.items():
    label_name = f"Cluster {cl}" if cl != -1 else "Noise / Outlier (-1)"
    print(f"  {label_name:25s}: {count:5d} ({count / len(labels):.2%})")

summary_rows = []
for cl in set(labels):
    sub = sample_df[sample_df['Cluster'] == cl]
    row = {'Cluster': 'Noise (-1)' if cl == -1 else f'Cluster {cl}', 'Count': len(sub)}
    for f in features:
        row[f] = round(sub[f].mean(), 2)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print("\n--- DBSCAN CLUSTER CLINICAL AVERAGES ---")
print(summary_df.to_string(index=False))

# Silhouette score on non-noise samples
non_noise_mask = (labels != -1)
if n_clusters > 1 and sum(non_noise_mask) > 10:
    sil = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
    print(f"\nSilhouette Score (Core/Border Points): {sil:.4f}")
else:
    sil = 0.0

# ============================================================
# 8. SAVE RESULTS (PLOTS & CSV)
# ============================================================
# 1. 2D PCA Projection Plot
pca = PCA(n_components=2, random_state=42)
pca_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
# Plot noise in gray
noise_pts = (labels == -1)
plt.scatter(
    pca_2d[noise_pts, 0], pca_2d[noise_pts, 1],
    c='#9ca3af', label='Noise / Outliers (-1)', alpha=0.4, s=20, marker='x'
)
# Plot clustered points
clustered_pts = (labels != -1)
scatter = plt.scatter(
    pca_2d[clustered_pts, 0], pca_2d[clustered_pts, 1],
    c=labels[clustered_pts], cmap='tab10', alpha=0.7, s=25, edgecolors='none'
)
plt.colorbar(scatter, label="DBSCAN Cluster ID")
plt.title(f"DBSCAN Density-Based Patient Clustering (Clusters: {n_clusters}, Noise: {n_noise})", fontsize=12, fontweight="bold")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

clusters_png = os.path.join(OUTPUT_FOLDER, "DBSCAN_Clusters.png")
plt.savefig(clusters_png, dpi=300)
plt.close()

# 2. Save Summary CSV
summary_csv = os.path.join(OUTPUT_FOLDER, "DBSCAN_Summary.csv")
summary_df.to_csv(summary_csv, index=False)

# 3. Save Metrics CSV
metrics_df = pd.DataFrame([{
    'Algorithm': 'DBSCAN',
    'eps': 1.2,
    'min_samples': 10,
    'Clusters_Formed': n_clusters,
    'Noise_Outliers': n_noise,
    'Noise_Percentage': round(n_noise / len(labels) * 100, 2),
    'Silhouette_Core_Points': round(sil, 4)
}])
metrics_csv = os.path.join(OUTPUT_FOLDER, "dbscan_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)

print(f"\n[OK] DBSCAN cluster plot saved to:    {clusters_png}")
print(f"[OK] DBSCAN profile summary saved to: {summary_csv}")
print(f"[OK] DBSCAN metrics saved to:         {metrics_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: DBSCAN CLUSTERING COMPLETED")
print("=" * 60)
