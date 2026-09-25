"""
=============================================================================
VITALSIGN: ORDINAL ENCODING DEMONSTRATION (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Feature: age brackets with natural ordered progression [0-10) to [90-100)
=============================================================================

ML Concept - Ordinal Encoding:
Ordinal encoding maps categorical categories to ordered integers when and only
when the variable possesses an inherent, meaningful mathematical ranking
(e.g., age groups, education tiers, disease severity).

CRITICAL CLINICAL DISTINCTION:
- Ordinal: 'age' ([0-10) < [10-20) < ... < [90-100)) represents chronological progression.
- Nominal: 'race', 'gender', 'admission_type_id' have NO mathematical ordering.
  Applying ordinal integers to nominal variables falsely injects artificial
  distances (e.g., claiming Category 3 > Category 1), which misleads algorithms.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

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
    print("VITALSIGN: ORDINAL ENCODING DEMONSTRATION (M2)")
    print("=" * 70)

    df = load_data()
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")

    # 1. Define explicit ordinal progression for Age
    # The UCI Diabetes dataset groups patient ages into 10-year brackets
    age_brackets_ordered = [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ]

    print("\n[1] Selected Ordinal Feature: 'age'")
    print(f"    Explicit Monotonic Order: {age_brackets_ordered}")

    # 2. Fit OrdinalEncoder with specified categories
    ordinal_encoder = OrdinalEncoder(
        categories=[age_brackets_ordered],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    encoded_age = ordinal_encoder.fit_transform(df[['age']])

    # 3. Create comparison DataFrame
    demo_df = pd.DataFrame({
        'encounter_id': df['encounter_id'].head(15),
        'age_raw': df['age'].head(15),
        'age_ordinal_encoded': encoded_age[:15].flatten().astype(int)
    })

    print("\n[2] Comparison of Raw Age Brackets vs Ordinal Encoded Integer:")
    print(demo_df.to_string(index=False))

    # Print category mapping
    print("\n[3] Explicit Mapping Table:")
    for idx, cat in enumerate(age_brackets_ordered):
        print(f"    Bracket {cat:10} -> Encoded Value: {idx}")

    # 4. Save demonstration CSV
    out_csv = OUTPUT_DIR / "clean_ordinal_encode_demo_M2.csv"
    demo_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Demonstration CSV saved to: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
