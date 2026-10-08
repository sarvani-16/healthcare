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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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
print("VITALSIGN - K-MEANS UNSUPERVISED CLUSTERING (M4)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (PHENOTYPING FEATURE SELECTION & SCALING)
# ============================================================
data = data.replace("?", np.nan)

features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_diagnoses'
]

print(f"Selected Clinical Numerical Phenotyping Features ({len(features)}):")
print(f"  {features}")

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(data[features])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ============================================================
# 6. MODEL / ANALYSIS (ELBOW METHOD & SILHOUETTE ANALYSIS)
# ============================================================
k_range = list(range(2, 11))
inertias = []
silhouettes = []

print("\nEvaluating Cluster Quality Across K = 2 to 10:")
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42)
    cluster_labels = kmeans_temp.fit_predict(X_scaled)

    inertia = kmeans_temp.inertia_
    inertias.append(inertia)

    # Subsample 5,000 encounters for fast silhouette computation
    sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=5000, random_state=42)
    silhouettes.append(sil_score)

    print(f"  K = {k:2d} | Inertia (WCSS): {inertia:12.2f} | Silhouette Score: {sil_score:.4f}")

# Train Selected Champion KMeans Model (K = 3 Clinical Phenotypes)
best_k = 3
print(f"\nFitting Final K-Means with Selected K = {best_k} (Clinical Phenotypes)...")
final_kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
final_labels = final_kmeans.fit_predict(X_scaled)

df_clustered = data[features].copy()
df_clustered['Cluster'] = final_labels

# ============================================================
# 7. EVALUATION / CLUSTER SUMMARY
# ============================================================
cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
print("\n--- PATIENT DISTRIBUTION ACROSS CLUSTERS ---")
for cl_id, count in cluster_counts.items():
    print(f"  Cluster {cl_id}: {count:,} patients ({count / len(data):.2%})")

cluster_profiles = df_clustered.groupby('Cluster').mean().round(2)
print("\n--- CLINICAL CLUSTER PROFILES (Feature Averages) ---")
print(cluster_profiles.to_string())

# ============================================================
# 8. SAVE RESULTS (PLOTS, METRICS & ARTIFACTS)
# ============================================================
# 1. Elbow Method Graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.title("K-Means Elbow Method for Optimal K (Inertia vs K)", fontsize=12, fontweight="bold")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.xticks(k_range)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
elbow_png = os.path.join(OUTPUT_FOLDER, "elbow_kmeans.png")
plt.savefig(elbow_png, dpi=300)
plt.close()

# 2. Silhouette Score Graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouettes, 'rs-', linewidth=2, markersize=8, color="#ef4444")
plt.title("Silhouette Score vs Number of Clusters (K)", fontsize=12, fontweight="bold")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Mean Silhouette Coefficient")
plt.xticks(k_range)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
sil_png = os.path.join(OUTPUT_FOLDER, "silhouette_scores.png")
plt.savefig(sil_png, dpi=300)
plt.close()

# 3. 2D Cluster Visualization via PCA
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled[:3000])

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    pca_coords[:, 0], pca_coords[:, 1],
    c=final_labels[:3000], cmap='viridis', alpha=0.6, edgecolors='none', s=25
)
plt.colorbar(scatter, label="Clinical Cluster ID")
plt.title(f"K-Means Patient Phenotypes (K={best_k}) - PCA Projection", fontsize=12, fontweight="bold")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
clusters_png = os.path.join(OUTPUT_FOLDER, "kmeans_clusters.png")
plt.savefig(clusters_png, dpi=300)
plt.close()

# 4. Save Cluster Evaluation CSV
k_eval_df = pd.DataFrame({
    'K_Clusters': k_range,
    'Inertia_WCSS': [round(x, 2) for x in inertias],
    'Silhouette_Score': [round(x, 4) for x in silhouettes]
})
k_eval_csv = os.path.join(OUTPUT_FOLDER, "kmeans_metrics.csv")
k_eval_df.to_csv(k_eval_csv, index=False)

# 5. Save Cluster Summary Profiles CSV
profile_csv = os.path.join(OUTPUT_FOLDER, "cluster_summary.csv")
cluster_profiles.to_csv(profile_csv)

# 6. Save Fitted Model
kmeans_model_path = os.path.join(MODELS_DIR, "kmeans_healthcare.pkl")
joblib.dump(final_kmeans, kmeans_model_path)

print(f"\n[OK] Elbow method plot saved to:       {elbow_png}")
print(f"[OK] Silhouette plot saved to:         {sil_png}")
print(f"[OK] 2D Cluster scatter saved to:      {clusters_png}")
print(f"[OK] Metrics table saved to:           {k_eval_csv}")
print(f"[OK] Cluster profile summary saved to: {profile_csv}")
print(f"[OK] Fitted KMeans model saved to:     {kmeans_model_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: K-MEANS CLUSTERING COMPLETED")
print("=" * 60)
