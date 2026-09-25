"""
=============================================================================
VITALSIGN: REAL-TIME PREDICTION TEST
Model: Healthcare/models/vitalsign_readmission_model.pkl
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
=============================================================================

ML Concept - Real-Time Inference:
Loads the serialized production pipeline, extracts a single patient encounter,
and evaluates patient risk without retraining.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
MODEL_PATH = BASE_DIR / "models" / "vitalsign_readmission_model.pkl"

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

DROP_COLUMNS = [
    'encounter_id', 'patient_nbr', 'readmitted',
    'weight', 'payer_code', 'medical_specialty'
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
    print("VITALSIGN: REAL-TIME PATIENT PREDICTION TEST")
    print("=" * 70)

    # 1. Check Model Existence
    if not MODEL_PATH.exists():
        print(f"[!] Saved model not found at {MODEL_PATH}.")
        print("    Running Random_Forest_Classifier_M3.py to train and save champion model...")
        import subprocess, sys
        rf_script = Path(__file__).parent / "Random_Forest_Classifier_M3.py"
        subprocess.run([sys.executable, str(rf_script)], check=True)

    print(f"\n[1] Loading Trained Deployment Pipeline from:\n    {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    # 2. Load dataset and select a sample encounter
    df = load_data()
    df['Readmission_30_Days'] = (df['readmitted'] == '<30').astype(int)

    # Pick sample patient (e.g. index 1)
    sample_idx = 1
    sample_patient = df.iloc[[sample_idx]].copy()
    actual_readmitted = sample_patient['readmitted'].values[0]
    actual_target = sample_patient['Readmission_30_Days'].values[0]

    # Prepare features by removing dropped and target columns
    features = sample_patient.drop(columns=[c for c in DROP_COLUMNS if c in sample_patient.columns] + ['Readmission_30_Days'])

    print(f"\n[2] Patient Encounter #{sample_patient['encounter_id'].values[0]} Clinical Profile:")
    print(f"    - Age Group:                 {sample_patient['age'].values[0]}")
    print(f"    - Gender:                    {sample_patient['gender'].values[0]}")
    print(f"    - Race:                      {sample_patient['race'].values[0]}")
    print(f"    - Time in Hospital:          {sample_patient['time_in_hospital'].values[0]} days")
    print(f"    - Lab Procedures:            {sample_patient['num_lab_procedures'].values[0]}")
    print(f"    - Procedures Performed:      {sample_patient['num_procedures'].values[0]}")
    print(f"    - Total Medications:         {sample_patient['num_medications'].values[0]}")
    print(f"    - Emergency Visits (prior):  {sample_patient['number_emergency'].values[0]}")
    print(f"    - Inpatient Visits (prior):  {sample_patient['number_inpatient'].values[0]}")
    print(f"    - Total Diagnoses:           {sample_patient['number_diagnoses'].values[0]}")
    print(f"    - Insulin Regimen:           {sample_patient['insulin'].values[0]}")

    # 3. Model Inference
    pred_class = pipeline.predict(features)[0]
    probabilities = pipeline.predict_proba(features)[0]
    prob_class1 = probabilities[1]

    # Clinical Risk Stratification
    if prob_class1 >= 0.60:
        risk_level = "HIGH RISK (Priority discharge protocol & follow-up recommended)"
    elif prob_class1 >= 0.30:
        risk_level = "MODERATE RISK (Telehealth check within 14 days suggested)"
    else:
        risk_level = "LOW RISK (Standard routine outpatient follow-up)"

    print("\n" + "=" * 50)
    print("PREDICTION RESULT:")
    print("=" * 50)
    print(f"Predicted 30-Day Readmission:")
    if pred_class == 1:
        print("  YES")
    else:
        print("  NO")

    print(f"\nProbability:")
    print(f"  {prob_class1 * 100:.2f}%")

    print(f"\nClinical Risk Stratification:")
    print(f"  {risk_level}")

    print(f"\nActual Hospital Outcome:")
    print(f"  Raw: '{actual_readmitted}' -> Binary Target: {actual_target}")
    print("-" * 50)
    print("DISCLAIMER: This application is an educational prediction")
    print("system and not a medical diagnosis tool.")
    print("=" * 70)

if __name__ == "__main__":
    main()
