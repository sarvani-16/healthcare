"""
=============================================================================
VITALSIGN: ADABOOST ENSEMBLE CLASSIFIER (M3)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: Readmission_30_Days (0 = No Readmit / >30d, 1 = Readmitted <30d)
=============================================================================

ML Concept - Adaptive Boosting (AdaBoost):
Trains an ensemble of weak base learners sequentially. At each boosting round m:
1. Samples misclassified by the previous weak learner receive higher sample weights:
      w_i := w_i * exp(alpha_m * I(y_i != h_m(x_i)))
2. A new base estimator is fitted to focus on previously difficult clinical encounters.
3. Final prediction combines all weak estimators weighted by their accuracy alpha_m:
      H(x) = sign(sum_{m=1}^M alpha_m * h_m(x))

ROBUST HYPERPARAMETER SELECTION:
Uses shallow decision tree estimators (max_depth=2) with learning_rate=0.5,
preventing estimator degradation on imbalanced healthcare datasets.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

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

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "AdaBoost"
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
    'weight', 'payer_code', 'medical_specialty',
    'diag_1', 'diag_2', 'diag_3'
]

def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)
    return df.replace("?", np.nan)

def main():
    print("=" * 70)
    print("VITALSIGN: ADABOOST ENSEMBLE CLASSIFIER (M3)")
    print("=" * 70)

    # 1. Load data and create target
    df = load_data()
    df['Readmission_30_Days'] = (df['readmitted'] == '<30').astype(int)
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")
    print(f"Target Distribution: Readmission Rate = {df['Readmission_30_Days'].mean():.2%}")

    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns] + ['Readmission_30_Days'])
    y = df['Readmission_30_Days']

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # 2. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Data Split: Train = {len(X_train):,} encounters, Test = {len(X_test):,} encounters")

    # 3. Construct Preprocessing & AdaBoost Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )

    base_tree = DecisionTreeClassifier(max_depth=2, random_state=42)

    ada_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=50,
            learning_rate=0.5,
            random_state=42
        ))
    ])

    print("\n[1] Training AdaBoost Classifier (50 Boosting Rounds, learning_rate=0.5)...")
    ada_pipeline.fit(X_train, y_train)

    # 4. Evaluation on test set
    print("\n[2] Evaluating Model on Holdout Test Set (10,000 Records)...")
    y_pred = ada_pipeline.predict(X_test)
    y_prob = ada_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n[3] PERFORMANCE METRICS:")
    print(f"    Accuracy:        {acc:.4f} ({acc:.2%})")
    print(f"    Precision:       {prec:.4f}")
    print(f"    Recall (Sens):   {rec:.4f} ({rec:.2%})")
    print(f"    F1-Score:        {f1:.4f}")
    print(f"    ROC-AUC Score:   {auc:.4f}")

    cls_report = classification_report(
        y_test, y_pred,
        target_names=["No Readmit / >30d (0)", "Readmitted <30d (1)"]
    )
    print("\n--- CLASSIFICATION REPORT ---")
    print(cls_report)

    # Save Classification Report
    report_file = OUTPUT_DIR / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("VITALSIGN ADABOOST ENSEMBLE CLASSIFIER REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
    print(f"[OK] Classification report saved to: {report_file}")

    # Save Metrics CSV
    metrics_df = pd.DataFrame([{
        'Model': 'AdaBoost Classifier (50 estimators, lr=0.5)',
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'ROC_AUC': round(auc, 4)
    }])
    metrics_file = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"[OK] Metrics table saved to: {metrics_file}")

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Oranges", cbar=False,
        xticklabels=["No Readmit (0)", "Readmitted (1)"],
        yticklabels=["No Readmit (0)", "Readmitted (1)"]
    )
    plt.title("AdaBoost Classifier Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Clinical Label", fontsize=10)
    plt.ylabel("Actual Clinical Label", fontsize=10)
    plt.tight_layout()

    cm_file = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=300)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {cm_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()