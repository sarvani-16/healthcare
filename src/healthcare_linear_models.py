"""
=============================================================================
VitalSign / HealthcarePrediction
Module: healthcare_linear_models.py
Fulfills CO2: Linear Supervised-Learning Models, Regularization & Scaling.
Models Demonstrated:
- Linear Regression (Ordinary Least Squares)
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization & Sparsity)
- Elastic Net Regression (Combined L1 + L2)
- Logistic Regression (Binary Classification on Readmission_30_Days)
- Multinomial Logistic Regression (Multiclass on Original 3-tier Readmission)
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

try:
    from .healthcare_dataset import load_dataset, clean_dataset, create_target
except ImportError:
    from healthcare_dataset import load_dataset, clean_dataset, create_target

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def demonstrate_feature_scaling():
    """
    Demonstrates StandardScaler vs MinMaxScaler on numeric clinical attributes:
    - StandardScaler: zero mean, unit variance (Gaussian assumption)
    - MinMaxScaler: bounds features strictly to [0, 1]
    """
    print("\n" + "=" * 60)
    print("CO2 DEMONSTRATION: FEATURE SCALING (Standard vs MinMax)")
    print("=" * 60)

    df = load_dataset()
    df = clean_dataset(df)

    num_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications']
    X_num = df[num_features].fillna(df[num_features].median())

    std_scaler = StandardScaler()
    mm_scaler = MinMaxScaler()

    X_std = std_scaler.fit_transform(X_num)
    X_mm = mm_scaler.fit_transform(X_num)

    print(f"Original Means:       {np.round(X_num.mean().values, 2)}")
    print(f"StandardScaled Means: {np.round(X_std.mean(axis=0), 2)} (Zero Mean)")
    print(f"StandardScaled Stds:  {np.round(X_std.std(axis=0), 2)} (Unit Variance)")
    print(f"MinMax Range:         [{X_mm.min():.1f}, {X_mm.max():.1f}] (Bounded [0, 1])")

def evaluate_regression_models():
    """
    Demonstrates Linear Regression, Ridge, Lasso, and Elastic Net
    on a suitable continuous clinical target: time_in_hospital (Length of Stay).
    Demonstrates regularization and the bias-variance trade-off.
    """
    print("\n" + "=" * 60)
    print("CO2 DEMONSTRATION: LINEAR REGRESSION & REGULARIZATION")
    print("Target: time_in_hospital (Continuous Days of Stay)")
    print("=" * 60)

    df = load_dataset()
    df = clean_dataset(df)

    # Predictors for length of stay
    feature_cols = [
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'
    ]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['time_in_hospital'].astype(float)

    # Scale predictors for regularized regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42
    )

    regressors = {
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "Ridge Regression (alpha=10.0)": Ridge(alpha=10.0, random_state=42),
        "Lasso Regression (alpha=0.01)": Lasso(alpha=0.01, random_state=42),
        "Lasso Regression (alpha=0.1)": Lasso(alpha=0.1, random_state=42),
        "Elastic Net (alpha=0.01, l1=0.5)": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        "Elastic Net (alpha=0.1, l1=0.5)": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    }

    results = []
    print(f"{'Model':<35} | {'MAE':<7} | {'MSE':<7} | {'RMSE':<7} | {'R2 Score':<8} | {'Zero Coefs':<10}")
    print("-" * 85)

    for name, model in regressors.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        zero_coefs = int(np.sum(np.abs(model.coef_) < 1e-4))

        results.append({
            "Task_Type": "Regression (Academic Demo: time_in_hospital)",
            "Model": name,
            "Metric_1": f"MAE: {mae:.4f}",
            "Metric_2": f"MSE: {mse:.4f}",
            "Metric_3": f"RMSE: {rmse:.4f}",
            "Primary_Score": f"R2: {r2:.4f}",
            "Notes": f"Zeroed Coefs: {zero_coefs}/{len(feature_cols)}"
        })
        print(f"{name:<35} | {mae:<7.4f} | {mse:<7.4f} | {rmse:<7.4f} | {r2:<8.4f} | {zero_coefs}/{len(feature_cols)}")

    print("-" * 85)
    print("\nBias-Variance Trade-off & Regularization Insights:")
    print("- OLS Linear Regression: Minimizes empirical training MSE without coefficient penalties (can suffer from multicollinearity).")
    print("- Ridge (L2 penalty): Shrinks regression coefficients towards zero, reducing variance at the cost of minor bias.")
    print("- Lasso (L1 penalty): Drives small coefficients exactly to zero, performing intrinsic clinical feature selection.")
    print("- Elastic Net: Combines L1 sparsity with L2 grouped-feature selection for robust continuous prediction.")

    return results

def evaluate_classification_linear_models():
    """
    Demonstrates Logistic Regression for binary readmission classification
    and Multinomial Logistic Regression on the original 3-tier target.
    """
    print("\n" + "=" * 60)
    print("CO2 DEMONSTRATION: LOGISTIC REGRESSION (Binary & Multinomial)")
    print("=" * 60)

    df = load_dataset()
    df = clean_dataset(df)

    num_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]
    X = df[num_cols].fillna(df[num_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    # 1. Binary Logistic Regression on Readmission_30_Days
    df_binary = create_target(df)
    y_binary = df_binary['Readmission_30_Days']

    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
        X_scaled, y_binary, test_size=0.20, random_state=42, stratify=y_binary
    )

    log_reg = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    log_reg.fit(X_tr_b, y_tr_b)
    y_pred_b = log_reg.predict(X_te_b)
    y_proba_b = log_reg.predict_proba(X_te_b)[:, 1]

    acc_b = accuracy_score(y_te_b, y_pred_b)
    prec_b = precision_score(y_te_b, y_pred_b, zero_division=0)
    rec_b = recall_score(y_te_b, y_pred_b, zero_division=0)
    f1_b = f1_score(y_te_b, y_pred_b, zero_division=0)
    auc_b = roc_auc_score(y_te_b, y_proba_b)

    print(f"\n[Binary Logistic Regression] Readmission_30_Days:")
    print(f"  Accuracy:  {acc_b:.4f}")
    print(f"  Precision: {prec_b:.4f}")
    print(f"  Recall:    {rec_b:.4f}")
    print(f"  F1-Score:  {f1_b:.4f}")
    print(f"  ROC-AUC:   {auc_b:.4f}")

    results.append({
        "Task_Type": "Binary Classification (Readmission_30_Days)",
        "Model": "Binary Logistic Regression (balanced)",
        "Metric_1": f"Accuracy: {acc_b:.4f}",
        "Metric_2": f"Precision: {prec_b:.4f}",
        "Metric_3": f"Recall: {rec_b:.4f}",
        "Primary_Score": f"F1: {f1_b:.4f}",
        "Notes": f"ROC-AUC: {auc_b:.4f}"
    })

    # 2. Multinomial Logistic Regression on 3-tier readmitted target ('<30', '>30', 'NO')
    y_multi = df['readmitted'].astype(str).str.strip()
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
        X_scaled, y_multi, test_size=0.20, random_state=42, stratify=y_multi
    )

    multi_log_reg = LogisticRegression(
        solver='lbfgs',
        max_iter=500,
        random_state=42
    )
    multi_log_reg.fit(X_tr_m, y_tr_m)
    y_pred_m = multi_log_reg.predict(X_te_m)

    acc_m = accuracy_score(y_te_m, y_pred_m)
    prec_m = precision_score(y_te_m, y_pred_m, average='weighted', zero_division=0)
    rec_m = recall_score(y_te_m, y_pred_m, average='weighted', zero_division=0)
    f1_m = f1_score(y_te_m, y_pred_m, average='weighted', zero_division=0)

    print(f"\n[Multinomial Logistic Regression] 3 Classes (<30, >30, NO):")
    print(f"  Accuracy:         {acc_m:.4f}")
    print(f"  Weighted Prec:    {prec_m:.4f}")
    print(f"  Weighted Recall:  {rec_m:.4f}")
    print(f"  Weighted F1:      {f1_m:.4f}")

    results.append({
        "Task_Type": "Multiclass Classification (Original readmitted 3-tier)",
        "Model": "Multinomial Logistic Regression (multiclass='multinomial')",
        "Metric_1": f"Accuracy: {acc_m:.4f}",
        "Metric_2": f"Weighted Precision: {prec_m:.4f}",
        "Metric_3": f"Weighted Recall: {rec_m:.4f}",
        "Primary_Score": f"Weighted F1: {f1_m:.4f}",
        "Notes": "Classes: <30, >30, NO"
    })

    return results

def run_all_linear_experiments():
    """Executes all CO2 linear experiments and exports outputs/linear_model_metrics.csv."""
    demonstrate_feature_scaling()
    reg_results = evaluate_regression_models()
    clf_results = evaluate_classification_linear_models()

    all_results = reg_results + clf_results
    df_metrics = pd.DataFrame(all_results)
    out_file = OUTPUTS_DIR / "linear_model_metrics.csv"
    df_metrics.to_csv(out_file, index=False)
    print(f"\n[OK] Linear model metrics exported to: {out_file}")
    return df_metrics

if __name__ == "__main__":
    print("=" * 70)
    print("VITALSIGN: healthcare_linear_models.py Execution")
    print("=" * 70)
    run_all_linear_experiments()
    print("\nCO2 Linear Models Analysis Completed Successfully.")
    print("=" * 70)
