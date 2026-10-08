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

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
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
print("VITALSIGN - HIERARCHICAL AGGLOMERATIVE CLUSTERING (M4)")
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

# Subsample 1,000 encounters for clear dendrogram rendering & pairwise distance matrix
sample_df = data[features].dropna().sample(n=1000, random_state=42)

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(sample_df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ============================================================
# 6. MODEL / ANALYSIS (WARD LINKAGE & AGGLOMERATIVE CLUSTERING)
# ============================================================
print("\n[1] Computing Ward Hierarchical Linkage Matrix...")
linked = linkage(X_scaled, method='ward')

print("[2] Fitting Agglomerative Clustering (n_clusters=3)...")
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster_labels = agg.fit_predict(X_scaled)
sample_df['Hierarchical_Cluster'] = cluster_labels

# ============================================================
# 7. EVALUATION / CLUSTER SUMMARY
# ============================================================
sil = silhouette_score(X_scaled, cluster_labels)
print(f"\nHierarchical Clustering Silhouette Score (K=3): {sil:.4f}")

cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print("\n--- PATIENT DISTRIBUTION ACROSS HIERARCHICAL CLUSTERS ---")
for cl, count in cluster_counts.items():
    print(f"  Cluster {cl}: {count:,} patients ({count / len(cluster_labels):.2%})")

summary_df = sample_df.groupby('Hierarchical_Cluster').mean().round(2)
print("\n--- CLINICAL HIERARCHICAL CLUSTER PROFILES ---")
print(summary_df.to_string())

# ============================================================
# 8. SAVE RESULTS (DENDROGRAM, SCATTER PLOT, SUMMARY CSV)
# ============================================================
# 1. Hierarchical Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(
    linked,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90.,
    leaf_font_size=9.,
    show_contracted=True
)
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=12, fontweight="bold")
plt.xlabel("Cluster Size / Encounter Index")
plt.ylabel("Ward Linkage Euclidean Distance")
plt.tight_layout()
dendro_png = os.path.join(OUTPUT_FOLDER, "hierarchical_dendrogram.png")
plt.savefig(dendro_png, dpi=300)
plt.close()

# 2. 2D PCA Cluster Scatter Plot
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    pca_coords[:, 0], pca_coords[:, 1],
    c=cluster_labels, cmap='viridis', alpha=0.7, s=30, edgecolors='none'
)
plt.colorbar(scatter, label="Hierarchical Cluster ID")
plt.title("Hierarchical Patient Clusters (K=3) - PCA Projection", fontsize=12, fontweight="bold")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
clusters_png = os.path.join(OUTPUT_FOLDER, "hierarchical_clusters.png")
plt.savefig(clusters_png, dpi=300)
plt.close()

# 3. Save Summary CSV
summary_csv = os.path.join(OUTPUT_FOLDER, "hierarchical_summary.csv")
summary_df.to_csv(summary_csv)

print(f"\n[OK] Dendrogram saved to:           {dendro_png}")
print(f"[OK] Cluster scatter plot saved to: {clusters_png}")
print(f"[OK] Summary profiles saved to:     {summary_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: HIERARCHICAL CLUSTERING COMPLETED")
print("=" * 60)
