"""
=============================================================================
VitalSign / HealthcarePrediction
Module 04: Machine Learning Model Training & Comparison
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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

COLUMNS_TO_REMOVE = [
    'encounter_id',
    'patient_nbr',
    'weight',
    'payer_code',
    'medical_specialty',
    'readmitted'
]

def load_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "readmitted" in first_line or "encounter_id" in first_line:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    df = df.replace("?", np.nan)
    return df

def main():
    print("=" * 70)
    print("VITALSIGN: 04_MODEL_TRAINING & COMPARISON")
    print("=" * 70)

    # 1. Load dataset
    df = load_data(DATASET_PATH)

    # 2. Target Creation (Readmission_30_Days)
    target_name = 'Readmission_30_Days'
    df[target_name] = (
        df['readmitted']
        .astype(str)
        .str.strip()
        .eq('<30')
        .astype(int)
    )

    # 3. Separate X and y without target leakage
    cols_to_drop = [c for c in COLUMNS_TO_REMOVE + [target_name] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_name]

    print(f"Features: {X.shape[1]} columns, Samples: {X.shape[0]}")
    print(f"Target Distribution: Class 1 (<30d) = {y.sum()} ({y.mean()*100:.2f}%), Class 0 = {(y==0).sum()}")

    # 4. train_test_split(test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 5. Load or Fit Preprocessor
    preprocessor_path = MODELS_DIR / "vitalsign_preprocessor.pkl"
    if preprocessor_path.exists():
        print("Loading fitted preprocessor...")
        preprocessor = joblib.load(preprocessor_path)
    else:
        print("Fitting ColumnTransformer preprocessor on training data...")
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder

        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ])
        preprocessor.fit(X_train)
        joblib.dump(preprocessor, preprocessor_path)

    print("Transforming feature matrices...")
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    print(f"Transformed dimension: {X_train_trans.shape[1]} features")

    # 6. Algorithms: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    metrics_list = []
    trained_models = {}

    print("\nTraining and evaluating candidate algorithms...")
    print("-" * 75)
    print(f"{'Model':<25} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9}")
    print("-" * 75)

    for name, clf in models.items():
        clf.fit(X_train_trans, y_train)
        y_pred = clf.predict(X_test_trans)
        y_proba = clf.predict_proba(X_test_trans)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4)
        })
        trained_models[name] = clf

        print(f"{name:<25} | {acc:<9.4f} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {roc_auc:<9.4f}")

    print("-" * 75)
    metrics_df = pd.DataFrame(metrics_list)

    # 7. Select best model based on F1-score
    best_row = metrics_df.sort_values(by="F1_Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest Model Selected (highest F1-score): {best_model_name}")
    print(f"  F1-Score: {best_row['F1_Score']:.4f}, Accuracy: {best_row['Accuracy']:.4f}, ROC-AUC: {best_row['ROC_AUC']:.4f}")

    # Mark selected in metrics
    metrics_df["Selected_Best"] = metrics_df["Model"] == best_model_name
    metrics_csv = OUTPUTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n[OK] Saved model comparison metrics to: {metrics_csv}")

    # 8. Save artifacts with required names:
    # models/vitalsign_model.pkl
    # models/vitalsign_preprocessor.pkl
    # models/vitalsign_metadata.pkl
    model_path = MODELS_DIR / "vitalsign_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"[OK] Saved Champion Model to:           {model_path}")

    joblib.dump(preprocessor, preprocessor_path)
    print(f"[OK] Saved Preprocessor to:             {preprocessor_path}")

    # Metadata
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    default_values = {}
    for col in numerical_cols:
        default_values[col] = float(X[col].median(skipna=True))
    for col in categorical_cols:
        mode_val = X[col].mode(dropna=True)
        default_values[col] = str(mode_val[0]) if len(mode_val) > 0 else "Unknown"

    metadata = {
        'best_model_name': best_model_name,
        'original_feature_columns': X.columns.tolist(),
        'feature_columns': X.columns.tolist(),
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'final_feature_columns': preprocessor.get_feature_names_out().tolist() if hasattr(preprocessor, "get_feature_names_out") else [],
        'target_name': target_name,
        'removed_columns': COLUMNS_TO_REMOVE,
        'default_values': default_values,
        'metrics': metrics_list,
        'best_metrics': best_row.to_dict(),
        'classes': [0, 1]
    }
    metadata_path = MODELS_DIR / "vitalsign_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"[OK] Saved Metadata to:                 {metadata_path}")

    print("\nModel training completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
