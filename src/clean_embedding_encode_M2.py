"""
=============================================================================
VITALSIGN: CATEGORICAL EMBEDDING ENCODING (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
=============================================================================

ML Concept - Categorical Embedding Representation:
In deep learning and neural network architectures, high-cardinality categorical
variables are represented via low-dimensional dense embedding vectors (similar to
Word2Vec). Prior to learning embeddings, categorical levels are mapped to integer
vocabulary indices [0, num_categories - 1], with an extra index reserved for
out-of-vocabulary or missing categories.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
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
    print("VITALSIGN: CATEGORICAL EMBEDDING VOCABULARY ENCODING (M2)")
    print("=" * 70)

    df = load_data()
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")

    selected_cols = ['race', 'gender', 'age', 'insulin', 'diabetesMed']
    subset = df[selected_cols].fillna('Missing').copy()

    encoded_dict = {}
    vocab_maps = {}

    for col in selected_cols:
        unique_vals = sorted(subset[col].unique().tolist())
        vocab = {val: idx for idx, val in enumerate(unique_vals)}
        vocab_maps[col] = vocab
        encoded_dict[f"{col}_embedding_idx"] = subset[col].map(vocab)

    encoded_df = pd.DataFrame(encoded_dict)
    combined_demo = pd.concat([subset.head(10), encoded_df.head(10)], axis=1)

    print("\n[1] Vocabulary Index Mappings for Neural Embeddings:")
    for col, vmap in vocab_maps.items():
        print(f"    - Feature '{col}' ({len(vmap)} unique levels): {vmap}")

    print("\n[2] Comparison of Categorical Strings vs Embedding Vocabulary Indices (First 10 Patients):")
    print(combined_demo[['race', 'race_embedding_idx', 'age', 'age_embedding_idx', 'insulin', 'insulin_embedding_idx']].to_string(index=False))

    out_csv = OUTPUT_DIR / "clean_embedded_encode_M2.csv"
    combined_demo.to_csv(out_csv, index=False)
    print(f"\n[OK] Embedding vocabulary demonstration saved to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
