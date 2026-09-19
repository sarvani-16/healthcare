"""
=============================================================================
VitalSign / HealthcarePrediction
Module: healthcare_model_training.py
Replaces and enhances M2 & M3 classification algorithms and comparisons.
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

try:
    from .healthcare_dataset import load_dataset
    from .healthcare_preprocessing import (
        prepare_features_target,
        build_preprocessing_pipeline,
        save_preprocessor,
        COLUMNS_TO_REMOVE
    )
except ImportError:
    from healthcare_dataset import load_dataset
    from healthcare_preprocessing import (
        prepare_features_target,
        build_preprocessing_pipeline,
        save_preprocessor,
        COLUMNS_TO_REMOVE
    )

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_compare_models(test_size: float = 0.20, random_state: int = 42):
    """
    Trains and compares:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. AdaBoost Classifier
    4. Random Forest Classifier
    Evaluates: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    Selects the best model based on F1-Score.
    """
    # 1. Load data
    df = load_dataset()
    X, y = prepare_features_target(df)

    # 2. Stratified train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Dataset split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows (stratified).")

    # 3. Fit preprocessing pipeline
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    print(f"Pipeline fitted. Transformed dimension: {X_train_trans.shape[1]} features.")

    # 4. Initialize candidate models
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=random_state
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=random_state
        ),
        "AdaBoostClassifier": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=50,
            learning_rate=0.5,
            random_state=random_state
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }

    metrics_list = []
    trained_clfs = {}

    print("\nTraining and evaluating candidate models:")
    print("-" * 75)
    print(f"{'Model':<25} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9}")
    print("-" * 75)

    for name, clf in candidates.items():
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
        trained_clfs[name] = clf
        print(f"{name:<25} | {acc:<9.4f} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {roc_auc:<9.4f}")

    print("-" * 75)
    metrics_df = pd.DataFrame(metrics_list)

    # 5. Select best model based on F1-Score
    best_row = metrics_df.sort_values(by="F1_Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = trained_clfs[best_name]

    print(f"\nBest Model Selected (highest F1-score): {best_name}")
    print(f"  F1-Score: {best_row['F1_Score']:.4f}, Accuracy: {best_row['Accuracy']:.4f}, ROC-AUC: {best_row['ROC_AUC']:.4f}")

    # Mark selected in metrics
    metrics_df["Selected_Best"] = metrics_df["Model"] == best_name
    metrics_csv = OUTPUTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[OK] Saved metrics to: {metrics_csv}")

    # 6. Save models and preprocessors with required names
    model_primary = MODELS_DIR / "vitalsign_readmission_model.pkl"
    model_alias = MODELS_DIR / "vitalsign_model.pkl"
    joblib.dump(best_model, model_primary)
    joblib.dump(best_model, model_alias)
    print(f"[OK] Saved Best Model to: {model_primary} and {model_alias}")

    save_preprocessor(preprocessor, "preprocessor.pkl")

    # Metadata dictionary
    default_vals = {}
    for col in numerical_cols:
        default_vals[col] = float(X[col].median(skipna=True))
    for col in categorical_cols:
        mode_val = X[col].mode(dropna=True)
        default_vals[col] = str(mode_val[0]) if len(mode_val) > 0 else "Unknown"

    metadata = {
        'feature_columns': X.columns.tolist(),
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'target_column': 'Readmission_30_Days',
        'removed_columns': COLUMNS_TO_REMOVE,
        'best_model_name': best_name,
        'class_names': ['No Readmission / >30 Days', 'Readmitted < 30 Days'],
        'default_values': default_vals,
        'metrics': metrics_list,
        'best_metrics': best_row.to_dict(),
        'classes': [0, 1]
    }

    meta_primary = MODELS_DIR / "model_metadata.pkl"
    meta_alias = MODELS_DIR / "vitalsign_metadata.pkl"
    joblib.dump(metadata, meta_primary)
    joblib.dump(metadata, meta_alias)
    print(f"[OK] Saved Metadata to: {meta_primary} and {meta_alias}")

    return best_model, preprocessor, metadata

if __name__ == "__main__":
    print("=" * 70)
    print("VITALSIGN: healthcare_model_training.py Execution")
    print("=" * 70)
    train_and_compare_models()
    print("Model training completed successfully.")
    print("=" * 70)
