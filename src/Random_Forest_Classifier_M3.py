"""
=============================================================================
VITALSIGN: RANDOM FOREST CLASSIFIER - CHAMPION MODEL (M3)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: Readmission_30_Days (0 = No Readmit / >30d, 1 = Readmitted <30d)
=============================================================================

ML Concept - Random Forest Ensemble:
Constructs an ensemble of decorrelated decision trees using bootstrap aggregation
(bagging) and random feature subspace projection (mtry = sqrt(p)).
    Prediction: Mode of individual tree classifications (majority vote)
    Probability: Average class probability across all B trees

CHAMPION MODEL SELECTION:
In imbalanced clinical prediction, individual decision trees suffer high variance.
Random Forest reduces variance exponentially without increasing bias, while
`class_weight='balanced'` ensures optimal recall for detecting vulnerable patients
facing 30-day readmission risk.

FLASK DEPLOYMENT INTEGRATION:
Serializes the fitted end-to-end pipeline to `models/vitalsign_readmission_model.pkl`,
directly powering real-time inference in Flask `app.py`.
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
OUTPUT_DIR = BASE_DIR / "outputs" / "Random_Forest"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
    print("VITALSIGN: RANDOM FOREST CLASSIFIER (CHAMPION MODEL) (M3)")
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

    # 2. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Data Split: Train = {len(X_train):,} encounters, Test = {len(X_test):,} encounters")

    # 3. Construct Preprocessing & Model Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )

    rf_pipeline = Pipeline([
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

    print("\n[1] Training Random Forest Classifier (100 Trees, class_weight='balanced')...")
    rf_pipeline.fit(X_train, y_train)

    # 4. Evaluation on test set
    print("\n[2] Evaluating Model on Holdout Test Set (10,000 Records)...")
    y_pred = rf_pipeline.predict(X_test)
    y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

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
        f.write("VITALSIGN RANDOM FOREST CLASSIFIER REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
    print(f"[OK] Classification report saved to: {report_file}")

    # Save Metrics CSV
    metrics_df = pd.DataFrame([{
        'Model': 'Random Forest Classifier (100 Trees, balanced)',
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
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
        xticklabels=["No Readmit (0)", "Readmitted (1)"],
        yticklabels=["No Readmit (0)", "Readmitted (1)"]
    )
    plt.title("Random Forest Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Clinical Label", fontsize=10)
    plt.ylabel("Actual Clinical Label", fontsize=10)
    plt.tight_layout()

    cm_file = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=300)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {cm_file}")

    # Top Feature Importances
    rf_model = rf_pipeline.named_steps['classifier']
    feature_names = num_cols + list(rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
    importances = rf_model.feature_importances_

    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    top15_feat = feat_df.head(15)
    print("\n--- TOP 10 CLINICAL PREDICTIVE FEATURES (Random Forest) ---")
    print(top15_feat.head(10).to_string(index=False))

    top15_feat.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=top15_feat, x='Importance', y='Feature', palette='mako')
    plt.title("Random Forest Top 15 Predictive Features", fontsize=12, fontweight="bold")
    plt.xlabel("Ensemble MDI Feature Importance", fontsize=10)
    plt.tight_layout()

    feat_png = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(feat_png, dpi=300)
    plt.close()
    print(f"[OK] Feature importance plot saved to: {feat_png}")

    # 5. Serialize Champion Model for Flask
    model_pkl = MODELS_DIR / "vitalsign_readmission_model.pkl"
    joblib.dump(rf_pipeline, model_pkl)
    print(f"\n[OK] Serialized Champion Model to: {model_pkl}")

    metadata = {
        'model_name': 'RandomForestClassifier',
        'n_estimators': 100,
        'features': list(X.columns),
        'metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc
        }
    }
    meta_pkl = MODELS_DIR / "model_metadata.pkl"
    joblib.dump(metadata, meta_pkl)
    print(f"[OK] Serialized Model Metadata to: {meta_pkl}")
    print("=" * 70)

if __name__ == "__main__":
    main()
