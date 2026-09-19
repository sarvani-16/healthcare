"""
=============================================================================
VitalSign / HealthcarePrediction
Module 06: Clinical Patient Prediction Test
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def get_risk_category(prob: float) -> str:
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Moderate Risk"
    else:
        return "High Risk"

def main():
    print("=" * 70)
    print("VITALSIGN: 06_PREDICTION_TEST")
    print("=" * 70)

    model_path = MODELS_DIR / "vitalsign_model.pkl"
    preprocessor_path = MODELS_DIR / "vitalsign_preprocessor.pkl"
    metadata_path = MODELS_DIR / "vitalsign_metadata.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        print("[ERROR] Required model artifacts missing. Please run 04_model_training.py first.")
        return

    # Load artifacts
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}

    print(f"Loaded Production Model: {type(model).__name__}")
    feature_cols = metadata.get("original_feature_columns", metadata.get("feature_columns", []))
    default_vals = metadata.get("default_values", {})

    # Create sample clinical encounter record using real dataset feature names
    sample_patient = {
        'race': 'Caucasian',
        'gender': 'Female',
        'age': '[60-70)',
        'admission_type_id': 1,               # Emergency
        'discharge_disposition_id': 1,        # Home
        'admission_source_id': 7,             # ER
        'time_in_hospital': 4,
        'num_lab_procedures': 45,
        'num_procedures': 1,
        'num_medications': 18,
        'number_outpatient': 0,
        'number_emergency': 1,
        'number_inpatient': 2,
        'diag_1': '250.83',
        'diag_2': '401',
        'diag_3': '272',
        'number_diagnoses': 8,
        'insulin': 'Steady',
        'change': 'Ch',
        'diabetesMed': 'Yes'
    }

    print("\n--- Input Patient Encounter Profile ---")
    for k, v in sample_patient.items():
        print(f"  {k:<26}: {v}")

    # Build complete input row using all expected feature columns
    full_row = {}
    for col in feature_cols:
        if col in sample_patient:
            full_row[col] = sample_patient[col]
        else:
            full_row[col] = default_vals.get(col, np.nan)

    input_df = pd.DataFrame([full_row])

    # Transform through fitted preprocessor
    X_trans = preprocessor.transform(input_df)

    # Predict class and probability
    predicted_class = int(model.predict(X_trans)[0])
    probabilities = model.predict_proba(X_trans)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    prob_30 = float(probabilities[1])

    risk_category = get_risk_category(prob_30)

    print("\n" + "=" * 70)
    print("PATIENT READMISSION RISK PREDICTION RESULT")
    print("=" * 70)
    print(f"Predicted Class:        {predicted_class} ({'Readmitted within 30 Days' if predicted_class == 1 else 'No Readmission within 30 Days'})")
    print(f"Prediction Probability: {prob_30:.4f} ({prob_30 * 100:.2f}%)")
    print(f"Risk Category:          {risk_category}")
    print("=" * 70)
    print("DISCLAIMER:")
    print("This application is an educational machine learning prediction system and is not a medical diagnosis tool. Please consult a qualified healthcare professional for medical decisions.")
    print("=" * 70)

if __name__ == "__main__":
    main()
