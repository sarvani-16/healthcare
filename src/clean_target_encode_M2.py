"""
=============================================================================
VITALSIGN: TARGET ENCODING DEMONSTRATION (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: Readmission_30_Days (Binary: 0 = No Readmit/>30d, 1 = Readmit <30d)
Feature: Categorical medical features ('race', 'admission_type_id')
=============================================================================

ML Concept - Target (Mean) Encoding:
Target encoding replaces each categorical level with the expected (mean) value
of the target variable for that category.

CRITICAL DATA LEAKAGE PREVENTION:
To prevent severe data leakage:
1. NEVER compute target statistics across the full dataset prior to splitting.
2. Split data into Train and Test sets FIRST.
3. Compute the conditional target mean ONLY on the training split:
      Target_Mean(c) = sum(y_train for c) / count(c in train)
4. Map these learned training means onto the test split.
5. Impute unseen or missing test categories with the global training mean (smoothing).
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    df = df.replace("?", np.nan)
    # Binary Readmission target: <30 -> 1, else 0
    df['Readmission_30_Days'] = (df['readmitted'] == '<30').astype(int)
    return df

def main():
    print("=" * 70)
    print("VITALSIGN: TARGET ENCODING DEMONSTRATION (M2)")
    print("=" * 70)

    df = load_data()
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")
    print(f"Target Distribution: Readmission Rate = {df['Readmission_30_Days'].mean():.2%}")

    feature_col = 'race'
    target_col = 'Readmission_30_Days'

    # Impute missing category for clean demo
    df[feature_col] = df[feature_col].fillna('Missing_Category')

    # STEP 1: Strict Train/Test Split (80/20 Stratified) to prevent data leakage
    print("\n[1] Splitting Dataset into Train (80%) and Test (20%) [Stratified]...")
    X_train, X_test, y_train, y_test = train_test_split(
        df[[feature_col, 'encounter_id']],
        df[target_col],
        test_size=0.20,
        random_state=42,
        stratify=df[target_col]
    )

    print(f"    Train Set: {len(X_train):,} samples | Test Set: {len(X_test):,} samples")

    # STEP 2: Calculate target mean statistics ONLY on training set
    global_train_mean = y_train.mean()
    print(f"\n[2] Computing Target Statistics on Train Split ONLY (Global Mean: {global_train_mean:.4f}):")

    train_stats = pd.DataFrame({
        'category': X_train[feature_col],
        'target': y_train
    }).groupby('category')['target'].agg(['count', 'mean']).rename(columns={'count': 'Train_Count', 'mean': 'Train_Target_Mean'})

    print(train_stats.to_string())

    # STEP 3: Map learned means onto Train and Test sets
    target_map = train_stats['Train_Target_Mean'].to_dict()

    X_train_encoded = X_train[feature_col].map(target_map).fillna(global_train_mean)
    X_test_encoded = X_test[feature_col].map(target_map).fillna(global_train_mean)

    # STEP 4: Demonstration comparison table
    demo_df = pd.DataFrame({
        'encounter_id': X_test['encounter_id'].head(15),
        'race_category': X_test[feature_col].head(15),
        'target_encoded_readmission_rate': X_test_encoded.head(15).round(4),
        'actual_y_test': y_test.head(15).values
    })

    print("\n[3] Test Set Target-Encoded Feature (First 15 Samples - Leakage Free):")
    print(demo_df.to_string(index=False))

    # Save demo CSV
    out_csv = OUTPUT_DIR / "clean_target_encode_demo_M2.csv"
    demo_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Target encoding demonstration CSV saved to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
