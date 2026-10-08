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
print("VITALSIGN - DECISION TREE CLASSIFIER (M3)")
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

# Fit preprocessor on training data
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# ============================================================
# 6. MODEL / ANALYSIS (TREE DEPTH COMPARISON & TRAINING)
# ============================================================
print("\n[1] Evaluating Decision Tree Max Depth Performance...")
depths = [3, 5, 8, 10, 15, None]
depth_labels = ['3', '5', '8', '10', '15', 'None (Full)']
train_f1s = []
test_f1s = []

for d in depths:
    dt_temp = DecisionTreeClassifier(
        max_depth=d,
        min_samples_split=20,
        class_weight='balanced',
        random_state=42
    )
    dt_temp.fit(X_train_trans, y_train)
    tr_pred = dt_temp.predict(X_train_trans)
    te_pred = dt_temp.predict(X_test_trans)
    train_f1s.append(f1_score(y_train, tr_pred))
    test_f1s.append(f1_score(y_test, te_pred))
    print(f"    Depth: {str(d):>10} | Train F1: {train_f1s[-1]:.4f} | Test F1: {test_f1s[-1]:.4f}")

# Train Best Decision Tree Model Pipeline (depth=10)
best_dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42
)

dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_dt)
])

print("\n[2] Training Best Decision Tree Pipeline (max_depth=10, min_samples_split=20)...")
dt_pipeline.fit(X_train, y_train)

# ============================================================
# 7. EVALUATION / PERFORMANCE METRICS
# ============================================================
y_pred = dt_pipeline.predict(X_test)
y_prob = dt_pipeline.predict_proba(X_test)[:, 1]

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
# 8. SAVE RESULTS (GRAPHS, METRICS, MODEL)
# ============================================================
# 1. Tree Depth Comparison Graph
plt.figure(figsize=(8, 5))
x_idx = np.arange(len(depths))
plt.plot(x_idx, train_f1s, 'o--', color='#2563eb', label='Train F1 Score (Fitting)')
plt.plot(x_idx, test_f1s, 's-', color='#dc2626', linewidth=2, label='Test F1 Score (Generalization)')
plt.xticks(x_idx, depth_labels)
plt.xlabel("Max Depth Hyperparameter", fontsize=10)
plt.ylabel("F1-Score", fontsize=10)
plt.title("Decision Tree Depth Comparison (Overfitting vs Generalization)", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
depth_png = os.path.join(OUTPUT_FOLDER, "Decision_Tree_Depth_Comparison.png")
plt.savefig(depth_png, dpi=300)
plt.close()

# 2. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Greens", cbar=False,
    xticklabels=["No Readmit (0)", "Readmitted (1)"],
    yticklabels=["No Readmit (0)", "Readmitted (1)"]
)
plt.title("Decision Tree Confusion Matrix", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Clinical Label")
plt.ylabel("Actual Clinical Label")
plt.tight_layout()
cm_png = os.path.join(OUTPUT_FOLDER, "Decision_Tree_Confusion_Matrix.png")
plt.savefig(cm_png, dpi=300)
plt.close()

# 3. Feature Importance Bar Plot
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
feature_names = num_cols + list(cat_encoder.get_feature_names_out(cat_cols))
importances = best_dt.feature_importances_

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top15_feat = feat_df.head(15)
plt.figure(figsize=(9, 5))
sns.barplot(data=top15_feat, x='Importance', y='Feature', palette='crest')
plt.title("Decision Tree Top 15 Predictive Features", fontsize=12, fontweight="bold")
plt.xlabel("Gini Feature Importance")
plt.tight_layout()
feat_png = os.path.join(OUTPUT_FOLDER, "Decision_Tree_Feature_Importance.png")
plt.savefig(feat_png, dpi=300)
plt.close()

# 4. Save Metrics CSV
metrics_df = pd.DataFrame([{
    'Model': 'Decision Tree (max_depth=10, balanced)',
    'Accuracy': round(acc, 4),
    'Precision': round(prec, 4),
    'Recall': round(rec, 4),
    'F1_Score': round(f1, 4),
    'ROC_AUC': round(auc, 4)
}])
metrics_csv = os.path.join(OUTPUT_FOLDER, "decision_tree_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)

# 5. Save Classification Report
report_txt = os.path.join(OUTPUT_FOLDER, "Decision_Tree_Classification_Report.txt")
with open(report_txt, "w", encoding="utf-8") as f:
    f.write("VITALSIGN DECISION TREE CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(cls_report)

# 6. Save Model Artifact
model_save_path = os.path.join(MODELS_DIR, "decision_tree_healthcare.pkl")
joblib.dump(dt_pipeline, model_save_path)

print(f"\n[OK] Depth comparison graph saved to: {depth_png}")
print(f"[OK] Confusion matrix saved to:        {cm_png}")
print(f"[OK] Feature importance plot saved to: {feat_png}")
print(f"[OK] Metrics table saved to:           {metrics_csv}")
print(f"[OK] Classification report saved to:   {report_txt}")
print(f"[OK] Trained model saved to:           {model_save_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: DECISION TREE CLASSIFIER COMPLETED")
print("=" * 60)