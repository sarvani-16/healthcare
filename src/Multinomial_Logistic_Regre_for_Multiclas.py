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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M2_Linear_Models"
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
    'encounter_id', 'patient_nbr', 'weight',
    'payer_code', 'medical_specialty',
    'diag_1', 'diag_2', 'diag_3'
]

# ============================================================
# 4. LOAD DATASET
# ============================================================
print("=" * 60)
print("VITALSIGN - MULTINOMIAL LOGISTIC REGRESSION (3-CLASS) (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (ORIGINAL 3-CLASS TARGET: NO, >30, <30)
# ============================================================
data = data.replace("?", np.nan)
target_col = 'readmitted'

X = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns] + [target_col])
y = data[target_col]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), cat_cols)
    ]
)

# ============================================================
# 6. MODEL / ANALYSIS (MULTINOMIAL SOFTMAX LBFGS)
# ============================================================
multi_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        solver='lbfgs',
        max_iter=300,
        random_state=42
    ))
])

print("Training Multinomial Softmax Logistic Regression...")
multi_pipeline.fit(X_train, y_train)

# ============================================================
# 7. EVALUATION / MULTICLASS METRICS
# ============================================================
y_pred = multi_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n--- MULTICLASS METRICS (MACRO-AVERAGED) ---")
print(f"Overall Accuracy:  {acc:.4f} ({acc:.2%})")
print(f"Macro Precision:   {prec:.4f}")
print(f"Macro Recall:      {rec:.4f}")
print(f"Macro F1-Score:    {f1:.4f}")

target_labels = ['NO', '>30', '<30']
cls_report = classification_report(y_test, y_pred, target_names=target_labels)
print("\n--- 3-CLASS CLASSIFICATION REPORT ---")
print(cls_report)

# ============================================================
# 8. SAVE RESULTS
# ============================================================
# Save text report
report_path = os.path.join(OUTPUT_FOLDER, "multinomial_classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("VITALSIGN MULTINOMIAL LOGISTIC REGRESSION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(cls_report)

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=target_labels)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Purples",
            xticklabels=target_labels, yticklabels=target_labels)
plt.title("Multinomial Logistic Regression Confusion Matrix", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Discharge Class")
plt.ylabel("Actual Discharge Class")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_FOLDER, "multinomial_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()

# Save metrics CSV
metrics_df = pd.DataFrame([{
    'Model': 'Multinomial Logistic Regression (3-Class)',
    'Accuracy': round(acc, 4),
    'Macro_Precision': round(prec, 4),
    'Macro_Recall': round(rec, 4),
    'Macro_F1': round(f1, 4)
}])
metrics_path = os.path.join(OUTPUT_FOLDER, "multinomial_logistic_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

# Save model artifact
model_path = os.path.join(MODELS_DIR, "multinomial_logistic_regression_healthcare.pkl")
joblib.dump(multi_pipeline, model_path)

print(f"\n[OK] Report saved to:           {report_path}")
print(f"[OK] Confusion matrix saved to: {cm_path}")
print(f"[OK] Metrics table saved to:    {metrics_path}")
print(f"[OK] Trained model saved to:    {model_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: MULTINOMIAL CLASSIFICATION COMPLETED")
print("=" * 60)
