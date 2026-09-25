"""
=============================================================================
VITALSIGN: CLOSED-FORM NORMAL EQUATION vs GRADIENT DESCENT (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: time_in_hospital (Continuous Length of Stay, in days)
Predictors: Clinical Numerical Attributes
=============================================================================

ML Concept - Closed-Form vs Iterative Optimization:
1. Closed-Form Normal Equation (CFNE):
   Analytically computes the exact optimal parameters beta in a single matrix
   operation without iterations:
       beta = (X^T * X)^(-1) * X^T * y
   Computationally exact, but requires O(p^3) matrix inversion, which becomes
   expensive when feature dimension p is huge.

2. Stochastic / Mini-Batch Gradient Descent (GD):
   Iteratively steps in the negative gradient direction of the loss surface:
       beta := beta - alpha * grad(Loss)
   Scales effectively to millions of samples and high dimensions, but requires
   feature standardization (StandardScaler) and hyperparameter tuning (learning rate alpha).
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data_50000.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
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
    print("VITALSIGN: NORMAL EQUATION (CFNE) vs GRADIENT DESCENT (GD) (M2)")
    print("=" * 70)

    df = load_data()
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")

    target_col = 'time_in_hospital'
    predictor_cols = [
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_diagnoses',
        'number_inpatient',
        'number_emergency'
    ]

    data = df[predictor_cols + [target_col]].dropna().copy()
    X = data[predictor_cols]
    y = data[target_col]

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Standardize predictors for Gradient Descent convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Closed-Form Normal Equation (via analytical OLS)
    print("\n[1] Fitting Closed-Form Normal Equation (OLS)...")
    cfne_model = LinearRegression()
    cfne_model.fit(X_train_scaled, y_train)
    y_pred_cfne = cfne_model.predict(X_test_scaled)

    cfne_mae = mean_absolute_error(y_test, y_pred_cfne)
    cfne_mse = mean_squared_error(y_test, y_pred_cfne)
    cfne_rmse = np.sqrt(cfne_mse)
    cfne_r2 = r2_score(y_test, y_pred_cfne)

    # 2. Gradient Descent (Iterative SGD Regressor)
    print("\n[2] Fitting Iterative Gradient Descent (SGDRegressor, max_iter=1000)...")
    gd_model = SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3, random_state=42)
    gd_model.fit(X_train_scaled, y_train)
    y_pred_gd = gd_model.predict(X_test_scaled)

    gd_mae = mean_absolute_error(y_test, y_pred_gd)
    gd_mse = mean_squared_error(y_test, y_pred_gd)
    gd_rmse = np.sqrt(gd_mse)
    gd_r2 = r2_score(y_test, y_pred_gd)

    # Comparison Table
    comparison_df = pd.DataFrame([
        {
            'Optimization_Method': 'Closed-Form Normal Equation (OLS)',
            'MAE': round(cfne_mae, 4),
            'MSE': round(cfne_mse, 4),
            'RMSE': round(cfne_rmse, 4),
            'R2_Score': round(cfne_r2, 4)
        },
        {
            'Optimization_Method': 'Stochastic Gradient Descent (SGD)',
            'MAE': round(gd_mae, 4),
            'MSE': round(gd_mse, 4),
            'RMSE': round(gd_rmse, 4),
            'R2_Score': round(gd_r2, 4)
        }
    ])

    print("\n[3] COMPARISON SUMMARY RESULTS:")
    print(comparison_df.to_string(index=False))

    # Save CSV
    out_csv = OUTPUT_DIR / "Linear_Regression_CFNE_GD_Comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Comparison CSV saved to: {out_csv}")

    # Plot Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Error Metrics
    x_indices = np.arange(2)
    width = 0.25
    axes[0].bar(x_indices - width/2, [cfne_mae, cfne_rmse], width, label="Normal Eq (CFNE)", color="#3b82f6")
    axes[0].bar(x_indices + width/2, [gd_mae, gd_rmse], width, label="Gradient Descent (GD)", color="#10b981")
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(['MAE (Days)', 'RMSE (Days)'], fontsize=11)
    axes[0].set_title("Error Metric Comparison (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: R2 Score
    axes[1].bar(["Normal Eq (CFNE)", "Gradient Descent (GD)"], [cfne_r2, gd_r2], color=["#3b82f6", "#10b981"], width=0.4)
    axes[1].set_ylim(0, max(cfne_r2, gd_r2) * 1.3)
    axes[1].set_ylabel("R-squared (R2)", fontsize=11)
    axes[1].set_title("Variance Explained (R2 Score)", fontsize=12, fontweight="bold")
    for i, v in enumerate([cfne_r2, gd_r2]):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("VitalSign: Normal Equation vs Gradient Descent Optimization", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_png = OUTPUT_DIR / "CFNE_vs_GD_Comparison.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] Comparison chart saved to: {out_png}")
    print("=" * 70)

if __name__ == "__main__":
    main()
