# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# 2. DATASET PATH
# ============================================================
DATASET_PATH = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

# ============================================================
# 3. OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/M2_Linear_Models"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

# ============================================================
# 4. LOAD DATASET
# ============================================================
print("=" * 60)
print("VITALSIGN - BIAS-VARIANCE TRADE-OFF ANALYSIS (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (NUMERICAL PREDICTORS & STRATIFIED SPLIT)
# ============================================================
data['Readmission_30_Days'] = (data['readmitted'] == '<30').astype(int)

# Use numerical features for clean, rapid complexity demonstration
num_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

X = data[num_cols].fillna(data[num_cols].median())
y = data['Readmission_30_Days']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ============================================================
# 6. MODEL / ANALYSIS (VARY TREE DEPTH COMPLEXITY)
# ============================================================
# ACADEMIC NOTE:
# - Low max_depth (e.g., 1, 2) -> High Bias (Underfitting): Model is too simple
#   to capture underlying clinical patterns; both train & val scores are poor.
# - High max_depth (e.g., None) -> High Variance (Overfitting): Model memorizes
#   noisy training instances; train score approaches 100%, but val score drops.

depth_candidates = [1, 2, 3, 5, 10, None]
depth_labels = ['1', '2', '3', '5', '10', 'None (Unlimited)']

train_accs = []
val_accs = []
train_f1s = []
val_f1s = []

print("\nEvaluating Decision Tree Complexity across Depth Hyperparameters:")
for depth, label in zip(depth_candidates, depth_labels):
    tree = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42)
    tree.fit(X_train, y_train)
    
    tr_pred = tree.predict(X_train)
    v_pred = tree.predict(X_val)
    
    tr_acc = accuracy_score(y_train, tr_pred)
    v_acc = accuracy_score(y_val, v_pred)
    tr_f1 = f1_score(y_train, tr_pred, zero_division=0)
    v_f1 = f1_score(y_val, v_pred, zero_division=0)
    
    train_accs.append(tr_acc)
    val_accs.append(v_acc)
    train_f1s.append(tr_f1)
    val_f1s.append(v_f1)
    
    print(f"  Depth {label:18} | Train Acc: {tr_acc:.4f} | Val Acc: {v_acc:.4f} | Train F1: {tr_f1:.4f} | Val F1: {v_f1:.4f}")

# ============================================================
# 7. EVALUATION / PLOTTING BIAS-VARIANCE CURVE
# ============================================================
x_indices = list(range(len(depth_candidates)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: Accuracy Curve
axes[0].plot(x_indices, train_accs, 'o-', color='#2563eb', lw=2, label='Training Accuracy')
axes[0].plot(x_indices, val_accs, 's--', color='#ef4444', lw=2, label='Validation Accuracy')
axes[0].set_xticks(x_indices)
axes[0].set_xticklabels(depth_labels)
axes[0].set_xlabel("Model Complexity (max_depth)", fontsize=11)
axes[0].set_ylabel("Accuracy", fontsize=11)
axes[0].set_title("Accuracy: Training vs Validation", fontsize=12, fontweight="bold")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Subplot 2: F1-Score Curve (Imbalance-Aware)
axes[1].plot(x_indices, train_f1s, 'o-', color='#2563eb', lw=2, label='Training F1-Score')
axes[1].plot(x_indices, val_f1s, 's--', color='#10b981', lw=2, label='Validation F1-Score')
axes[1].set_xticks(x_indices)
axes[1].set_xticklabels(depth_labels)
axes[1].set_xlabel("Model Complexity (max_depth)", fontsize=11)
axes[1].set_ylabel("F1-Score", fontsize=11)
axes[1].set_title("F1-Score: Bias vs Variance Trade-off", fontsize=12, fontweight="bold")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle("VitalSign: Bias-Variance Trade-off Analysis (Underfitting vs Overfitting)", fontsize=13, fontweight="bold")
plt.tight_layout()

chart_path = os.path.join(OUTPUT_FOLDER, "bias_variance.png")
plt.savefig(chart_path, dpi=300)
plt.close()

# ============================================================
# 8. SAVE RESULTS
# ============================================================
summary_df = pd.DataFrame({
    'max_depth': depth_labels,
    'Train_Accuracy': [round(x, 4) for x in train_accs],
    'Validation_Accuracy': [round(x, 4) for x in val_accs],
    'Train_F1': [round(x, 4) for x in train_f1s],
    'Validation_F1': [round(x, 4) for x in val_f1s],
    'Diagnosis': [
        'High Bias (Underfitting)',
        'Underfitting',
        'Balanced Generalization',
        'Optimal Depth (Sweet Spot)',
        'Beginning Overfitting',
        'High Variance (Severe Overfitting)'
    ]
})

csv_path = os.path.join(OUTPUT_FOLDER, "bias_variance_summary.csv")
summary_df.to_csv(csv_path, index=False)

print(f"\n[OK] Bias-variance plot saved to:    {chart_path}")
print(f"[OK] Bias-variance summary saved to: {csv_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: BIAS-VARIANCE ANALYSIS COMPLETED")
print("=" * 60)
