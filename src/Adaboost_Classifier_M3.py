# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M3_Tree_Models"
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
print("=" * 60)
print("VITALSIGN - ADABOOST ENSEMBLE CLASSIFIER (M3)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (TARGET CREATION & STRATIFIED SPLIT)
# ============================================================
data = data.replace("?", np.nan)
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

X = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns] + ['Readmission_30_Days'])
y = data['Readmission_30_Days']

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), cat_cols)
    ]
)

# ============================================================
# 6. MODEL / ANALYSIS (ADABOOST ENSEMBLE PIPELINE)
# ============================================================
base_tree = DecisionTreeClassifier(max_depth=2, random_state=42)

ada_classifier = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

ada_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', ada_classifier)
])

print("\nTraining AdaBoost Ensemble (50 Boosting Rounds, learning_rate=0.5)...")
ada_pipeline.fit(X_train, y_train)

# ============================================================
# 7. EVALUATION / PERFORMANCE METRICS
# ============================================================
y_pred = ada_pipeline.predict(X_test)
y_prob = ada_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n--- PERFORMANCE METRICS ---")
print(f"Accuracy:        {acc:.4f} ({acc:.2%})")
print(f"Precision:       {prec:.4f}")
print(f"Recall (Sens):   {rec:.4f} ({rec:.2%})")
print(f"F1-Score:        {f1:.4f}")
print(f"ROC-AUC Score:   {auc:.4f}")

cls_report = classification_report(
    y_test, y_pred,
    target_names=["No Readmission / >30d (0)", "Readmitted <30d (1)"]
)
print("\n--- CLASSIFICATION REPORT ---")
print(cls_report)

# ============================================================
# 8. SAVE RESULTS (GRAPHS, METRICS & MODEL)
# ============================================================
# 1. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Oranges", cbar=False,
    xticklabels=["No Readmit (0)", "Readmitted (1)"],
    yticklabels=["No Readmit (0)", "Readmitted (1)"]
)
plt.title("AdaBoost Classifier Confusion Matrix", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Clinical Label")
plt.ylabel("Actual Clinical Label")
plt.tight_layout()
cm_png = os.path.join(OUTPUT_FOLDER, "AdaBoost_Confusion_Matrix.png")
plt.savefig(cm_png, dpi=300)
plt.close()

# 2. Save Metrics CSV
metrics_df = pd.DataFrame([{
    'Model': 'AdaBoost Classifier (50 estimators, lr=0.5)',
    'Accuracy': round(acc, 4),
    'Precision': round(prec, 4),
    'Recall': round(rec, 4),
    'F1_Score': round(f1, 4),
    'ROC_AUC': round(auc, 4)
}])
metrics_csv = os.path.join(OUTPUT_FOLDER, "adaboost_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)

# 3. Save Classification Report
report_txt = os.path.join(OUTPUT_FOLDER, "AdaBoost_Classification_Report.txt")
with open(report_txt, "w", encoding="utf-8") as f:
    f.write("VITALSIGN ADABOOST CLASSIFIER REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(cls_report)

# 4. Save Model Artifact
model_save_path = os.path.join(MODELS_DIR, "adaboost_healthcare.pkl")
joblib.dump(ada_pipeline, model_save_path)

print(f"\n[OK] Confusion matrix saved to:        {cm_png}")
print(f"[OK] Metrics table saved to:           {metrics_csv}")
print(f"[OK] Classification report saved to:   {report_txt}")
print(f"[OK] Model saved to:                   {model_save_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: ADABOOST ENSEMBLE CLASSIFIER COMPLETED")
print("=" * 60)