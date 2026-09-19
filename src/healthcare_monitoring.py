"""
=============================================================================
VitalSign / HealthcarePrediction
Module: healthcare_monitoring.py
Production inference monitoring, audit logging, drift concepts, and risk tracking.
Logs predictions to outputs/prediction_log.csv.
=============================================================================
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUTS_DIR / "prediction_log.csv"

LOG_COLUMNS = [
    "timestamp",
    "predicted_class",
    "probability",
    "risk_category",
    "model_name",
    "time_in_hospital",
    "num_medications",
    "num_lab_procedures",
    "number_diagnoses",
    "age_group",
    "gender"
]

def initialize_log_file():
    """Ensures prediction log CSV exists with appropriate headers."""
    if not LOG_FILE.exists():
        df_empty = pd.DataFrame(columns=LOG_COLUMNS)
        df_empty.to_csv(LOG_FILE, index=False)
        print(f"[OK] Initialized monitoring audit log: {LOG_FILE}")

def log_prediction(patient_input: dict, prediction_result: dict, model_name: str = "RandomForestClassifier"):
    """
    Logs a single clinical prediction event to outputs/prediction_log.csv.
    Stores anonymized summary clinical attributes without direct personal identifiers.
    """
    initialize_log_file()

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_class": prediction_result.get("prediction", 0),
        "probability": round(float(prediction_result.get("probability", 0.0)), 4),
        "risk_category": prediction_result.get("risk_category", "Low Risk"),
        "model_name": model_name,
        "time_in_hospital": patient_input.get("time_in_hospital", 0),
        "num_medications": patient_input.get("num_medications", 0),
        "num_lab_procedures": patient_input.get("num_lab_procedures", 0),
        "number_diagnoses": patient_input.get("number_diagnoses", 0),
        "age_group": str(patient_input.get("age", "[60-70)")),
        "gender": str(patient_input.get("gender", "Female"))
    }

    df_entry = pd.DataFrame([entry])
    df_entry.to_csv(LOG_FILE, mode="a", header=False, index=False)
    return entry

def get_monitoring_statistics() -> dict:
    """
    Computes real-time inference statistics from prediction logs:
    - Total predictions
    - Low-risk predictions
    - Moderate-risk predictions
    - High-risk predictions
    - Average prediction probability
    """
    initialize_log_file()

    if not LOG_FILE.exists() or os.path.getsize(LOG_FILE) == 0:
        return {
            "total_predictions": 0,
            "low_risk_count": 0,
            "moderate_risk_count": 0,
            "high_risk_count": 0,
            "avg_probability": 0.0,
            "risk_distribution": {}
        }

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return {
            "total_predictions": 0,
            "low_risk_count": 0,
            "moderate_risk_count": 0,
            "high_risk_count": 0,
            "avg_probability": 0.0,
            "risk_distribution": {}
        }

    total = len(df)
    low = int((df["risk_category"] == "Low Risk").sum())
    mod = int((df["risk_category"] == "Moderate Risk").sum())
    high = int((df["risk_category"] == "High Risk").sum())
    avg_prob = round(float(df["probability"].mean()), 4)

    return {
        "total_predictions": total,
        "low_risk_count": low,
        "moderate_risk_count": mod,
        "high_risk_count": high,
        "avg_probability": avg_prob,
        "risk_distribution": {
            "Low Risk": low,
            "Moderate Risk": mod,
            "High Risk": high
        }
    }

def print_mlops_monitoring_guide():
    """Prints academic conceptual explanations of monitoring, drift, and retraining."""
    print("\n" + "=" * 75)
    print("ACADEMIC MLOPS MONITORING & GOVERNANCE CONCEPTS")
    print("=" * 75)
    print("""
1. DATA DRIFT (Covariate Shift):
   - Definition: Change in the statistical distribution of predictor features P(X)
     over time while conditional probability P(Y|X) remains constant.
   - Clinical Example: Shift in patient population demographics (e.g. higher median age
     or new diagnostic admission patterns post-pandemic).
   - Detection: Two-sample Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI).

2. CONCEPT DRIFT:
   - Definition: Change in the relationship between patient features and readmission outcomes P(Y|X).
   - Clinical Example: Introduction of a hospital-wide diabetic case-management program
     reduces readmissions despite high patient severity.
   - Mitigation: Continuous outcome verification, rolling retraining windows.

3. PREDICTION DISTRIBUTION MONITORING:
   - Tracks proportion of Low, Moderate, and High Risk predictions.
   - A sharp surge in High-Risk flags signals potential feature skew or covariate shift.

4. TRAINING-SERVING SKEW:
   - Discrepancy between how data is processed during offline training vs real-time inference.
   - VitalSign Solution: Reuses the identical ColumnTransformer (preprocessor.pkl) across both
     batch training and Flask single-patient inference.

5. MODEL RETRAINING TRIGGER:
   - Automated or threshold-based retraining executed when F1-score decays or PSI > 0.25.
    """)
    print("=" * 75)

if __name__ == "__main__":
    print("=" * 70)
    print("VITALSIGN: healthcare_monitoring.py Execution")
    print("=" * 70)

    # Demonstrate logging sample predictions
    sample_inputs = [
        {"time_in_hospital": 2, "num_medications": 8, "num_lab_procedures": 25, "number_diagnoses": 4, "age": "[40-50)", "gender": "Male"},
        {"time_in_hospital": 7, "num_medications": 24, "num_lab_procedures": 65, "number_diagnoses": 9, "age": "[70-80)", "gender": "Female"},
        {"time_in_hospital": 12, "num_medications": 32, "num_lab_procedures": 80, "number_diagnoses": 11, "age": "[80-90)", "gender": "Female"}
    ]
    sample_preds = [
        {"prediction": 0, "probability": 0.18, "risk_category": "Low Risk"},
        {"prediction": 1, "probability": 0.54, "risk_category": "Moderate Risk"},
        {"prediction": 1, "probability": 0.78, "risk_category": "High Risk"}
    ]

    for inp, pred in zip(sample_inputs, sample_preds):
        logged = log_prediction(inp, pred)
        print(f"Logged Prediction Event: {logged['risk_category']} (Prob: {logged['probability']})")

    stats = get_monitoring_statistics()
    print("\nCurrent Monitoring Statistics:")
    print(f"  Total Predictions:          {stats['total_predictions']}")
    print(f"  Low-Risk Predictions:       {stats['low_risk_count']}")
    print(f"  Moderate-Risk Predictions:  {stats['moderate_risk_count']}")
    print(f"  High-Risk Predictions:      {stats['high_risk_count']}")
    print(f"  Average Probability:        {stats['avg_probability']:.4f}")

    print_mlops_monitoring_guide()
    print("Monitoring module verification completed.")
    print("=" * 70)
