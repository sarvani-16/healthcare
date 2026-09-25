"""
=============================================================================
VITALSIGN: MULTINOMIAL LOGISTIC REGRESSION FOR 3-CLASS TARGET (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: readmitted (3-Class Categorical: 'NO', '>30', '<30')
=============================================================================

ML Concept - Multinomial Logistic Regression (Softmax Regression):
Extends binary logistic regression to K > 2 mutually exclusive classes.
The model calculates K linear functions and applies the Softmax function to
obtain normalized class probabilities:
    P(Y = k | X) = exp(beta_k^T * X) / sum_{j=1}^K exp(beta_j^T * X)

CRITICAL ACADEMIC CONTEXT:
This script provides an educational demonstration of multiclass categorization
on the original 3 hospital discharge categories ('NO', '>30', '<30').
The primary project and clinical policy target remains binary 30-day readmission.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "Multinomial_Logistic_Regression"
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

# Exclude identifiers, sparse columns, and ultra-high cardinality raw diagnostic codes
DROP_COLUMNS = [
    'encounter_id', 'patient_nbr', 'weight',
    'payer_code', 'medical_specialty',
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
    print("VITALSIGN: MULTINOMIAL LOGISTIC REGRESSION (3-CLASS) (M2)")
    print("=" * 70)

    df = load_data()
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")

    # 3-Class Target: readmitted ('NO', '>30', '<30')
    target_col = 'readmitted'
    class_counts = df[target_col].value_counts()
    print(f"\n[1] 3-Class Target Distribution ('{target_col}'):")
    for cls_name, cnt in class_counts.items():
        print(f"    - Class '{cls_name:5}': {cnt:,} encounters ({cnt / len(df):.2%})")

    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns] + [target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2] Stratified Split: Train = {len(X_train):,}, Test = {len(X_test):,}")

    # Build Pipeline
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

    # Note: Modern scikit-learn uses solver='lbfgs' directly for multinomial softmax
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            solver='lbfgs',
            max_iter=300,
            random_state=42
        ))
    ])

    print("\n[3] Training Multinomial Logistic Regression Pipeline...")
    clf_pipeline.fit(X_train, y_train)

    print("\n[4] Evaluating on Holdout Test Set (10,000 Records)...")
    y_pred = clf_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n[5] MULTICLASS PERFORMANCE METRICS (Macro-Averaged):")
    print(f"    Overall Accuracy:      {acc:.4f} ({acc:.2%})")
    print(f"    Macro Precision:       {prec:.4f}")
    print(f"    Macro Recall:          {rec:.4f}")
    print(f"    Macro F1-Score:        {f1:.4f}")

    target_labels = ['NO', '>30', '<30']
    cls_report = classification_report(y_test, y_pred, target_names=target_labels)
    print("\n--- 3-CLASS CLASSIFICATION REPORT ---")
    print(cls_report)

    # Save Classification Report
    report_file = OUTPUT_DIR / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("VITALSIGN MULTINOMIAL LOGISTIC REGRESSION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
    print(f"[OK] Report saved to: {report_file}")

    # Save Metrics CSV
    metrics_df = pd.DataFrame([{
        'Model': 'Multinomial Logistic Regression (3-Class)',
        'Accuracy': round(acc, 4),
        'Macro_Precision': round(prec, 4),
        'Macro_Recall': round(rec, 4),
        'Macro_F1': round(f1, 4)
    }])
    metrics_file = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"[OK] Metrics table saved to: {metrics_file}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=target_labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Purples",
        xticklabels=target_labels, yticklabels=target_labels
    )
    plt.title("Multinomial Logistic Regression Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Readmission Class", fontsize=10)
    plt.ylabel("Actual Readmission Class", fontsize=10)
    plt.tight_layout()

    cm_file = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=300)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {cm_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
