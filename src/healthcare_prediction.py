"""
=============================================================================
VitalSign / HealthcarePrediction
Module: healthcare_prediction.py
Handles single-patient and batch inference with clinical risk scoring.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

DISCLAIMER_TEXT = (
    "This application is an educational prediction system and not a medical diagnosis tool. "
    "Please consult a qualified healthcare professional for medical decisions."
)

_CACHED_MODEL = None
_CACHED_PREPROCESSOR = None
_CACHED_METADATA = None

def load_model_and_metadata(force_reload: bool = False):
    """
    Loads and caches the champion trained model, preprocessing pipeline, and metadata.
    Supports both vitalsign_readmission_model.pkl and vitalsign_model.pkl.
    """
    global _CACHED_MODEL, _CACHED_PREPROCESSOR, _CACHED_METADATA

    if not force_reload and _CACHED_MODEL is not None and _CACHED_PREPROCESSOR is not None:
        return _CACHED_MODEL, _CACHED_PREPROCESSOR, _CACHED_METADATA

    # Model path resolution
    model_path = MODELS_DIR / "vitalsign_readmission_model.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "vitalsign_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found in {MODELS_DIR}. Run training first.")

    # Preprocessor path resolution
    prep_path = MODELS_DIR / "preprocessor.pkl"
    if not prep_path.exists():
        prep_path = MODELS_DIR / "vitalsign_preprocessor.pkl"
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor pipeline not found in {MODELS_DIR}. Run training first.")

    # Metadata path resolution
    meta_path = MODELS_DIR / "model_metadata.pkl"
    if not meta_path.exists():
        meta_path = MODELS_DIR / "vitalsign_metadata.pkl"

    _CACHED_MODEL = joblib.load(model_path)
    _CACHED_PREPROCESSOR = joblib.load(prep_path)
    _CACHED_METADATA = joblib.load(meta_path) if meta_path.exists() else {}

    return _CACHED_MODEL, _CACHED_PREPROCESSOR, _CACHED_METADATA

def get_risk_tier(prob: float):
    """
    Classifies 30-day readmission risk into tiers:
    - Low Risk: prob < 0.30
    - Moderate Risk: 0.30 <= prob < 0.60
    - High Risk: prob >= 0.60
    """
    if prob < 0.30:
        return {
            "tier": "Low Risk",
            "badge_class": "success",
            "action": "Routine outpatient follow-up recommended.",
            "color_hex": "#10b981"
        }
    elif prob < 0.60:
        return {
            "tier": "Moderate Risk",
            "badge_class": "warning",
            "action": "Enhanced discharge planning and 14-day telemedicine check suggested.",
            "color_hex": "#f59e0b"
        }
    else:
        return {
            "tier": "High Risk",
            "badge_class": "danger",
            "action": "Priority clinical intervention, post-discharge nurse visit, and medication reconciliation advised.",
            "color_hex": "#ef4444"
        }

def prepare_patient_row(patient_dict: dict, feature_cols: list, default_vals: dict) -> pd.DataFrame:
    """
    Aligns patient dictionary input to the exact feature columns required by the model.
    Fills missing values with training set defaults or NaN for imputer processing.
    """
    row = {}
    for col in feature_cols:
        val = patient_dict.get(col, None)
        if val is None or val == "" or str(val).strip() == "?":
            row[col] = default_vals.get(col, np.nan)
        else:
            try:
                s_val = str(val).strip()
                if s_val.replace(".", "", 1).isdigit() or (s_val.startswith("-") and s_val[1:].replace(".", "", 1).isdigit()):
                    row[col] = float(s_val) if "." in s_val else int(s_val)
                else:
                    row[col] = s_val
            except Exception:
                row[col] = val

    return pd.DataFrame([row])

def predict_patient(patient_dict: dict) -> dict:
    """
    Predicts 30-day readmission risk for a single patient dictionary.
    Returns structured results including prediction, probability, risk tier, and educational disclaimer.
    """
    from sklearn.pipeline import Pipeline

    model, preprocessor, metadata = load_model_and_metadata()
    feature_cols = metadata.get("feature_columns") or metadata.get("features") or [
        'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
        'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
        'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
        'glyburide_metformin', 'glipizide_metformin', 'glimepiride_pioglitazone',
        'metformin_rosiglitazone', 'metformin_pioglitazone', 'change', 'diabetesMed'
    ]
    default_vals = metadata.get("default_values", {})

    input_df = prepare_patient_row(patient_dict, feature_cols, default_vals)

    # Check if model is an end-to-end Pipeline with built-in preprocessor
    if isinstance(model, Pipeline):
        pred_class = int(model.predict(input_df)[0])
        if hasattr(model, "predict_proba"):
            prob_array = model.predict_proba(input_df)[0]
            prob_readmit = float(prob_array[1])
        else:
            prob_readmit = 1.0 if pred_class == 1 else 0.0
    else:
        X_trans = preprocessor.transform(input_df)
        pred_class = int(model.predict(X_trans)[0])
        if hasattr(model, "predict_proba"):
            prob_array = model.predict_proba(X_trans)[0]
            prob_readmit = float(prob_array[1])
        else:
            prob_readmit = 1.0 if pred_class == 1 else 0.0

    risk_info = get_risk_tier(prob_readmit)

    return {
        "prediction": pred_class,
        "prediction_label": "Readmitted within 30 Days" if pred_class == 1 else "No Readmission / >30 Days",
        "probability": round(prob_readmit, 4),
        "probability_percent": round(prob_readmit * 100, 2),
        "risk_category": risk_info["tier"],
        "risk_badge": risk_info["badge_class"],
        "risk_color": risk_info["color_hex"],
        "recommendation": risk_info["action"],
        "model_used": metadata.get("best_model_name", type(model).__name__),
        "disclaimer": DISCLAIMER_TEXT
    }

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts readmission for a batch pandas DataFrame.
    Returns input DataFrame with prediction columns appended.
    """
    from sklearn.pipeline import Pipeline

    model, preprocessor, metadata = load_model_and_metadata()
    feature_cols = metadata.get("feature_columns") or metadata.get("features") or [
        'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
        'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
        'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
        'glyburide_metformin', 'glipizide_metformin', 'glimepiride_pioglitazone',
        'metformin_rosiglitazone', 'metformin_pioglitazone', 'change', 'diabetesMed'
    ]
    default_vals = metadata.get("default_values", {})

    rows = []
    for _, record in df.iterrows():
        record_dict = record.to_dict()
        row = {}
        for col in feature_cols:
            val = record_dict.get(col, None)
            if val is None or pd.isna(val) or val == "" or str(val).strip() == "?":
                row[col] = default_vals.get(col, np.nan)
            else:
                row[col] = val
        rows.append(row)

    aligned_df = pd.DataFrame(rows)

    if isinstance(model, Pipeline):
        preds = model.predict(aligned_df)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(aligned_df)[:, 1]
        else:
            probas = preds.astype(float)
    else:
        X_trans = preprocessor.transform(aligned_df)
        preds = model.predict(X_trans)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_trans)[:, 1]
        else:
            probas = preds.astype(float)

    result_df = df.copy()
    result_df["Predicted_Readmission_30d"] = preds
    result_df["Readmission_Probability"] = np.round(probas, 4)
    result_df["Risk_Category"] = [get_risk_tier(p)["tier"] for p in probas]

    return result_df

if __name__ == "__main__":
    print("=" * 70)
    print("VITALSIGN: healthcare_prediction.py Execution")
    print("=" * 70)

    sample_patient = {
        'race': 'Caucasian',
        'gender': 'Female',
        'age': '[60-70)',
        'admission_type_id': 1,
        'discharge_disposition_id': 1,
        'admission_source_id': 7,
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

    result = predict_patient(sample_patient)
    print("Patient Assessment Result:")
    for k, v in result.items():
        print(f"  {k:<22}: {v}")
    print("=" * 70)
