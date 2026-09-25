"""
=============================================================================
VITALSIGN: END-TO-END MACHINE LEARNING LIFECYCLE DEMO
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: Readmission_30_Days (Binary: 0 = No Readmit / >30d, 1 = Readmitted <30d)
Model: Random Forest Ensemble (Deployment Champion)
=============================================================================

ML Concept - Complete Production Machine Learning Lifecycle:
Demonstrates the full 8-stage operational journey from raw hospital records
to a validated, serialized model delivering real-time clinical inference:

  [1] DATA INGESTION
          │
  [2] DATA CLEANING & MISSING TOKEN RESOLUTION
          │
  [3] FEATURE ENGINEERING & TARGET DERIVATION
          │
  [4] LEAKAGE-FREE STRATIFIED TRAIN/TEST SPLIT
          │
  [5] PIPELINE INTEGRATION & MODEL TRAINING
          │
  [6] MULTI-METRIC PERFORMANCE EVALUATION
          │
  [7] MODEL SERIALIZATION & METADATA REGISTRATION
          │
  [8] REAL-TIME CLINICAL PREDICTION INFERENCE
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
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

DROP_COLUMNS = [
    'encounter_id', 'patient_nbr', 'readmitted',
    'weight', 'payer_code', 'medical_specialty'
]

def main():
    print("=" * 70)
    print("VITALSIGN: COMPLETE END-TO-END ML LIFECYCLE DEMONSTRATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1: DATA INGESTION
    # -------------------------------------------------------------
    print("\n[STAGE 1: DATA INGESTION]")
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "readmitted" in first_line or "encounter_id" in first_line:
        raw_df = pd.read_csv(DATASET_PATH)
    else:
        raw_df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)
    print(f" -> Ingested {len(raw_df):,} raw clinical encounters with {raw_df.shape[1]} attributes.")

    # -------------------------------------------------------------
    # STAGE 2: DATA CLEANING
    # -------------------------------------------------------------
    print("\n[STAGE 2: DATA CLEANING & MISSING TOKEN RESOLUTION]")
    clean_df = raw_df.replace("?", np.nan)
    dups = clean_df.duplicated().sum()
    if dups > 0:
        clean_df = clean_df.drop_duplicates()
    print(f" -> Converted '?' missing tokens to NaN. Duplicate records removed: {dups}.")

    # -------------------------------------------------------------
    # STAGE 3: FEATURE PREPARATION & TARGET DERIVATION
    # -------------------------------------------------------------
    print("\n[STAGE 3: FEATURE PREPARATION & TARGET DERIVATION]")
    clean_df['Readmission_30_Days'] = (clean_df['readmitted'] == '<30').astype(int)
    pos_count = (clean_df['Readmission_30_Days'] == 1).sum()
    print(f" -> Derived binary target 'Readmission_30_Days' (<30 -> 1, else 0).")
    print(f" -> Class Distribution: {pos_count:,} Readmitted ({pos_count / len(clean_df):.2%}), {len(clean_df) - pos_count:,} Not Readmitted.")

    X = clean_df.drop(columns=[c for c in DROP_COLUMNS if c in clean_df.columns] + ['Readmission_30_Days'])
    y = clean_df['Readmission_30_Days']

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f" -> Features: {len(num_cols)} numerical, {len(cat_cols)} categorical. Dropped leakage-prone IDs.")

    # -------------------------------------------------------------
    # STAGE 4: LEAKAGE-FREE TRAIN/TEST SPLIT
    # -------------------------------------------------------------
    print("\n[STAGE 4: LEAKAGE-FREE STRATIFIED TRAIN/TEST SPLIT]")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f" -> Split: Train Set = {len(X_train):,} encounters (80%), Test Set = {len(X_test):,} encounters (20%).")

    # -------------------------------------------------------------
    # STAGE 5: PIPELINE INTEGRATION & MODEL TRAINING
    # -------------------------------------------------------------
    print("\n[STAGE 5: PIPELINE INTEGRATION & MODEL TRAINING]")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )

    champion_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    print(" -> Fitting Random Forest Classifier (100 Trees, class_weight='balanced')...")
    champion_pipeline.fit(X_train, y_train)
    print(" -> Model training complete.")

    # -------------------------------------------------------------
    # STAGE 6: MULTI-METRIC PERFORMANCE EVALUATION
    # -------------------------------------------------------------
    print("\n[STAGE 6: MULTI-METRIC PERFORMANCE EVALUATION]")
    y_pred = champion_pipeline.predict(X_test)
    y_prob = champion_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f" -> Test Accuracy:       {acc:.2%}")
    print(f" -> Test Precision:      {prec:.4f}")
    print(f" -> Test Recall (Sens):  {rec:.2%} (Identifies vulnerable readmission cases)")
    print(f" -> Test F1-Score:       {f1:.4f}")
    print(f" -> Test ROC-AUC:        {auc:.4f}")

    # -------------------------------------------------------------
    # STAGE 7: MODEL SERIALIZATION
    # -------------------------------------------------------------
    print("\n[STAGE 7: MODEL SERIALIZATION & DEPLOYMENT PREPARATION]")
    model_path = MODELS_DIR / "vitalsign_readmission_model.pkl"
    joblib.dump(champion_pipeline, model_path)
    print(f" -> Champion model saved to: {model_path}")

    meta = {
        'model_name': 'RandomForestClassifier',
        'n_estimators': 100,
        'features': list(X.columns),
        'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'roc_auc': auc}
    }
    joblib.dump(meta, MODELS_DIR / "model_metadata.pkl")
    print(f" -> Deployment metadata saved to: {MODELS_DIR / 'model_metadata.pkl'}")

    # -------------------------------------------------------------
    # STAGE 8: REAL-TIME INFERENCE SIMULATION
    # -------------------------------------------------------------
    print("\n[STAGE 8: REAL-TIME CLINICAL PREDICTION INFERENCE]")
    sample_patient = X_test.iloc[0:1].copy()
    actual_label = y_test.iloc[0]

    predicted_class = champion_pipeline.predict(sample_patient)[0]
    predicted_prob = champion_pipeline.predict_proba(sample_patient)[0, 1]

    readmit_str = "YES (High Risk of 30-Day Readmission)" if predicted_class == 1 else "NO (Low Risk of 30-Day Readmission)"

    if predicted_prob >= 0.60:
        risk_level = "High Risk"
    elif predicted_prob >= 0.30:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    print("\n" + "=" * 50)
    print("CLINICAL RISK PREDICTION RESULT:")
    print("=" * 50)
    print(f"Patient Record Sample Feature Summary:")
    print(f"  - Age Bracket:       {sample_patient['age'].values[0]}")
    print(f"  - Time in Hospital:  {sample_patient['time_in_hospital'].values[0]} days")
    print(f"  - Lab Procedures:    {sample_patient['num_lab_procedures'].values[0]}")
    print(f"  - Medications:       {sample_patient['num_medications'].values[0]}")
    print(f"  - Inpatient Visits:  {sample_patient['number_inpatient'].values[0]}")
    print("--------------------------------------------------")
    print(f"Predicted 30-Day Readmission: {readmit_str}")
    print(f"Readmission Probability:      {predicted_prob:.2%}")
    print(f"Clinical Risk Stratification: {risk_level}")
    print(f"Actual Clinical Outcome:      {'Readmitted (<30d)' if actual_label == 1 else 'Not Readmitted'}")
    print("--------------------------------------------------")
    print("DISCLAIMER: This application is an educational prediction")
    print("system and not a medical diagnosis tool.")
    print("=" * 70)

if __name__ == "__main__":
    main()
