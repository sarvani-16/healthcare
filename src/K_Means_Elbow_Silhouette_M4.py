"""
=============================================================================
VITALSIGN: K-MEANS UNSUPERVISED CLUSTERING (M4)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Task: Unsupervised Clinical Patient Phenotyping / Segmentation
Methods: Elbow Method (Inertia) & Silhouette Analysis (K = 2 to 10)
=============================================================================

ML Concept - Unsupervised K-Means Clustering:
Partitions N clinical encounters into K distinct clusters where each patient
belongs to the cluster with the nearest mean (cluster centroid):
    Inertia (WCSS) = sum_{i=1}^N min_{mu_k} || x_i - mu_k ||^2

CLUSTER VALIDATION:
1. Elbow Method: Plots inertia across K. The "elbow" inflection marks diminishing
   returns in variance reduction.
2. Silhouette Coefficient s(i) = (b(i) - a(i)) / max(a(i), b(i)) in [-1, +1]:
   Quantifies intra-cluster cohesion vs nearest-cluster separation.

CRITICAL ACADEMIC DISTINCTION:
K-Means is an UNSUPERVISED segmentation technique. It does NOT predict
Readmission_30_Days directly. Instead, it identifies clinical patient phenotypes
(e.g., routine stay vs complex multi-diagnostic hospitalization).
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "KMeans"
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

def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)
    return df.replace("?", np.nan)

def main():
    print("=" * 70)
    print("VITALSIGN: K-MEANS CLUSTERING (ELBOW & SILHOUETTE) (M4)")
    print("=" * 70)

    df = load_data()
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")

    # 1. Select continuous clinical features for patient segmentation
    features = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_diagnoses'
    ]
    print(f"\n[1] Selected Numerical Phenotyping Features ({len(features)}):")
    print(f"    {features}")

    # 2. Impute and scale
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df[features])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 3. Test K values from 2 to 10
    k_range = list(range(2, 11))
    inertias = []
    silhouettes = []

    print("\n[2] Evaluating Cluster Quality Across K = 2 to 10...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Inertia (Within-Cluster Sum of Squares)
        inertia = kmeans.inertia_
        inertias.append(inertia)

        # Silhouette score computed on representative 5,000 patient sample for speed
        sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=5000, random_state=42)
        silhouettes.append(sil_score)

        print(f"    K = {k:2d} | Inertia (WCSS): {inertia:12.2f} | Silhouette Score: {sil_score:.4f}")

    # Summary table
    k_eval_df = pd.DataFrame({
        'K_Clusters': k_range,
        'Inertia_WCSS': [round(x, 2) for x in inertias],
        'Silhouette_Score': [round(x, 4) for x in silhouettes]
    })
    k_eval_df.to_csv(OUTPUT_DIR / "kmeans_k_evaluation.csv", index=False)

    # 4. Generate Elbow Method Plot
    print("\n[3] Generating Elbow Method Graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.title("K-Means Elbow Method for Optimal K (Inertia vs K)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=11)
    plt.xticks(k_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    elbow_png = OUTPUT_DIR / "Elbow_Method.png"
    plt.savefig(elbow_png, dpi=300)
    plt.close()
    print(f"    [OK] Saved: {elbow_png}")

    # 5. Generate Silhouette Score Plot
    print("\n[4] Generating Silhouette Score Graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouettes, 'rs-', linewidth=2, markersize=8, color="#ef4444")
    plt.title("Silhouette Score vs Number of Clusters (K)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Mean Silhouette Coefficient", fontsize=11)
    plt.xticks(k_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    sil_png = OUTPUT_DIR / "Silhouette_Score.png"
    plt.savefig(sil_png, dpi=300)
    plt.close()
    print(f"    [OK] Saved: {sil_png}")

    # 6. Fit Final KMeans with selected K=3
    best_k = 3
    print(f"\n[5] Fitting Final KMeans with Selected K = {best_k} (Clinical Phenotypes)...")
    final_kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
    final_labels = final_kmeans.fit_predict(X_scaled)

    df_clustered = df[features].copy()
    df_clustered['Cluster'] = final_labels

    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    print("\n--- PATIENT DISTRIBUTION ACROSS CLUSTERS ---")
    for cl_id, count in cluster_counts.items():
        print(f"    Cluster {cl_id}: {count:,} patients ({count / len(df):.2%})")

    # Cluster Profiles (Mean values of clinical features)
    cluster_profiles = df_clustered.groupby('Cluster').mean().round(2)
    print("\n--- CLINICAL CLUSTER PROFILES (Feature Averages) ---")
    print(cluster_profiles.to_string())

    profile_csv = OUTPUT_DIR / "KMeans_Cluster_Profiles.csv"
    cluster_profiles.to_csv(profile_csv)
    print(f"\n[OK] Cluster profiles saved to: {profile_csv}")

    # 7. 2D Cluster Visualization via PCA
    print("\n[6] Generating 2D Cluster Visualization via PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled[:3000])

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=final_labels[:3000], cmap='viridis', alpha=0.6, edgecolors='none', s=25
    )
    plt.colorbar(scatter, label="Clinical Cluster ID")
    plt.title(f"K-Means Patient Phenotypes (K={best_k}) - PCA Projection", fontsize=13, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=10)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    clusters_png = OUTPUT_DIR / "KMeans_Clusters.png"
    plt.savefig(clusters_png, dpi=300)
    plt.close()
    print(f"    [OK] Saved: {clusters_png}")
    print("=" * 70)

if __name__ == "__main__":
    main()
