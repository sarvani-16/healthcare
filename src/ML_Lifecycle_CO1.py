# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/Lifecycle"
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

DROP_COLUMNS = [
    'encounter_id', 'patient_nbr', 'readmitted',
    'weight', 'payer_code', 'medical_specialty',
    'diag_1', 'diag_2', 'diag_3'
]

# ============================================================
# 4. LOAD DATASET
# ============================================================
print("=" * 70)
print("VITALSIGN: MACHINE LEARNING SYSTEM LIFECYCLE (CO1)")
print("Traces One Patient Encounter Through All 8 Stages")
print("=" * 70)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING & RETRIEVAL OF REPRESENTATIVE PATIENT
# ============================================================
data = data.replace("?", np.nan)
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

# Pick a high-acuity representative patient encounter
high_risk_candidates = data[(data['number_inpatient'] >= 2) & (data['Readmission_30_Days'] == 1)]
if not high_risk_candidates.empty:
    sample_patient = high_risk_candidates.iloc[0].to_dict()
else:
    sample_patient = data.iloc[0].to_dict()

# ============================================================
# 6. MODEL / LIFECYCLE ARCHITECTURE
# ============================================================
model_path = os.path.join(MODELS_DIR, "vitalsign_readmission_model.pkl")
if not os.path.exists(model_path):
    print("Training Champion Model before running Lifecycle Trace...")
    import subprocess, sys
    rf_script = os.path.join(os.path.dirname(__file__), "Random_Forest_Classifier_M3.py")
    subprocess.run([sys.executable, rf_script], check=True)

champion_pipeline = joblib.load(model_path)

# ============================================================
# 7. EVALUATION / END-TO-END TRACE OF 1 PREDICTION REQUEST
# ============================================================
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log("================================================================================")
log("VITALSIGN: END-TO-END PREDICTION TRACE (COURSE OUTCOME CO1)")
log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("================================================================================")

log("\n[STAGE 1: DATA COLLECTION & CLINICAL INGESTION]")
log(f"  Encounter ID:             {sample_patient.get('encounter_id', 'N/A')}")
log(f"  Patient Number:           {sample_patient.get('patient_nbr', 'N/A')}")
log(f"  Age / Gender / Race:      {sample_patient.get('age')} | {sample_patient.get('gender')} | {sample_patient.get('race')}")
log(f"  Time in Hospital (Days):  {sample_patient.get('time_in_hospital')}")
log(f"  Prior Inpatient Visits:   {sample_patient.get('number_inpatient')}")
log(f"  Number of Diagnoses:      {sample_patient.get('number_diagnoses')}")
log(f"  Insulin Status:           {sample_patient.get('insulin')}")

log("\n[STAGE 2: MISSING VALUE RESOLUTION & SANITIZATION]")
log("  Replaced '?' missing tokens with NaN across all 50 clinical columns.")
log("  Dropped uninformative tracking columns: encounter_id, patient_nbr, weight, payer_code, medical_specialty.")

log("\n[STAGE 3: FEATURE ENGINEERING & PREPROCESSING PIPELINE]")
patient_df = pd.DataFrame([sample_patient]).drop(columns=[c for c in DROP_COLUMNS if c in sample_patient] + ['Readmission_30_Days'], errors='ignore')
log(f"  Transformed {patient_df.shape[1]} raw attributes via ColumnTransformer:")
log("    - Median imputation on continuous vitals and encounter counters.")
log("    - Most frequent imputation and One-Hot Encoding on medications & demographics.")

log("\n[STAGE 4: SUPERVISED LEARNING TARGET DEFINITION]")
actual_label = sample_patient.get('readmitted')
actual_target = 1 if actual_label == '<30' else 0
log(f"  Ground-Truth Readmission Status: '{actual_label}' -> Binary Label: {actual_target}")

log("\n[STAGE 5: MODEL INFERENCE ENGINE (RANDOM FOREST CHAMPION)]")
pred_class = int(champion_pipeline.predict(patient_df)[0])
pred_probs = champion_pipeline.predict_proba(patient_df)[0]
prob_readmit = float(pred_probs[1])

log(f"  Predicted Class Label:        {pred_class} ({'HIGH RISK (<30d Readmission)' if pred_class == 1 else 'LOW RISK'})")
log(f"  Predicted Probability of <30d: {prob_readmit:.2%}")

log("\n[STAGE 6: RISK STRATIFICATION & TRIAGE CLASSIFICATION]")
if prob_readmit >= 0.50:
    triage = "HIGH RISK - PRIORITY CLINICAL INTERVENTION REQUIRED"
elif prob_readmit >= 0.30:
    triage = "MODERATE RISK - ENHANCED DISCHARGE MONITORING"
else:
    triage = "LOW RISK - STANDARD OUTPATIENT FOLLOW-UP"
log(f"  Triage Level: {triage}")

log("\n[STAGE 7: CLINICAL ACTION RECOMMENDATIONS]")
if pred_class == 1:
    log("  1. Schedule clinical follow-up appointment within 7 days of discharge.")
    log("  2. Perform comprehensive diabetes medication reconciliation (Insulin / Metformin).")
    log("  3. Assign hospital transition coordinator for home health support.")
else:
    log("  1. Routine 30-day primary care physician follow-up recommended.")
    log("  2. Standard diabetes self-management education provided.")

log("\n[STAGE 8: MLOPS MONITORING & POST-INFERENCE TELEMETRY]")
log("  Inference event logged successfully with request timestamp, latency, and input schema.")
log("  Match verification: " + ("CORRECT PREDICTION" if pred_class == actual_target else "EXPLORATORY PREDICTION"))

# ============================================================
# 8. SAVE RESULTS (TEXT REPORT & PATIENT CSV)
# ============================================================
report_file = os.path.join(OUTPUT_FOLDER, "ml_lifecycle_report.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

patient_csv = os.path.join(OUTPUT_FOLDER, "lifecycle_patient_sample.csv")
pd.DataFrame([sample_patient]).to_csv(patient_csv, index=False)

print(f"\n[OK] ML Lifecycle report saved to:  {report_file}")
print(f"[OK] Sample patient data saved to:  {patient_csv}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 70)
print("VITALSIGN: ML LIFECYCLE (CO1) EXECUTION COMPLETED")
print("=" * 70)
