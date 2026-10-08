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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/Model_Comparison"
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
print("VITALSIGN - MULTI-MODEL BENCHMARK & COMPARISON (M3)")
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
# 6. MODEL / ANALYSIS (BENCHMARK 4 ALGORITHMS)
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=20, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42),
    'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=42), n_estimators=50, learning_rate=0.5, random_state=42)
}

results = []
fitted_pipelines = {}

print("\nTraining and Evaluating Candidate Models on Holdout Test Set:")
for name, clf in models.items():
    print(f"  Fitting {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipeline.fit(X_train, y_train)
    fitted_pipelines[name] = pipeline

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'ROC_AUC': round(auc, 4)
    })

# ============================================================
# 7. EVALUATION / BENCHMARK SUMMARY TABLE
# ============================================================
comp_df = pd.DataFrame(results)
print("\n--- COMPREHENSIVE BENCHMARK RESULTS TABLE ---")
print(comp_df.to_string(index=False))

# Identify Champion Model by F1-Score
champion_idx = comp_df['F1_Score'].idxmax()
champion_name = comp_df.loc[champion_idx, 'Model']
champion_f1 = comp_df.loc[champion_idx, 'F1_Score']
champion_rec = comp_df.loc[champion_idx, 'Recall']

print(f"\nCHAMPION MODEL SELECTED: {champion_name}")
print(f"  - F1-Score: {champion_f1:.4f}")
print(f"  - Recall:   {champion_rec:.2%}")

# ============================================================
# 8. SAVE RESULTS (CSV, BAR PLOT & METADATA)
# ============================================================
# 1. Save Comparison CSV
out_csv = os.path.join(OUTPUT_FOLDER, "model_comparison.csv")
comp_df.to_csv(out_csv, index=False)

# 2. Multi-Metric Comparison Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
x = np.arange(len(comp_df['Model']))
width = 0.15

plt.figure(figsize=(12, 6))
colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    offset = (i - len(metrics) / 2) * width + width / 2
    plt.bar(x + offset, comp_df[metric], width, label=metric, color=color)

plt.xticks(x, comp_df['Model'], fontsize=11, fontweight='bold')
plt.ylabel('Score (0.0 - 1.0)', fontsize=11)
plt.title('VitalSign: Machine Learning Model Comparison on 30-Day Readmission', fontsize=13, fontweight='bold')
plt.legend(loc='upper right', framealpha=0.9)
plt.ylim(0, 1.0)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()

out_png = os.path.join(OUTPUT_FOLDER, "model_comparison.png")
plt.savefig(out_png, dpi=300)
plt.close()

# 3. Update Champion Model Metadata
champ_row = comp_df.loc[champion_idx]
metadata = {
    'champion_model': champion_name,
    'metrics': {
        'accuracy': float(champ_row['Accuracy']),
        'precision': float(champ_row['Precision']),
        'recall': float(champ_row['Recall']),
        'f1_score': float(champ_row['F1_Score']),
        'roc_auc': float(champ_row['ROC_AUC'])
    },
    'all_models': results
}
meta_pkl = os.path.join(MODELS_DIR, "model_metadata.pkl")
joblib.dump(metadata, meta_pkl)

print(f"\n[OK] Comparison CSV saved to:      {out_csv}")
print(f"[OK] Comparison bar plot saved to: {out_png}")
print(f"[OK] Model metadata saved to:      {meta_pkl}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: MULTI-MODEL BENCHMARK COMPLETED")
print("=" * 60)
