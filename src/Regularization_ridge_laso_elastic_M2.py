"""
=============================================================================
VITALSIGN: REGULARIZATION - RIDGE, LASSO, & ELASTIC NET (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: time_in_hospital (Continuous Length of Stay, in days)
Predictors: Standardized Clinical Numerical Attributes
=============================================================================

ML Concept - Regularization Techniques:
1. Ridge Regression (L2 Regularization):
   Adds quadratic penalty: Loss + lambda * sum(beta_j^2).
   Shrinks coefficients towards zero, handles multicollinearity, keeps all features.

2. Lasso Regression (L1 Regularization):
   Adds absolute value penalty: Loss + lambda * sum(|beta_j|).
   Drives non-informative feature weights strictly to zero (automatic feature selection).

3. Elastic Net (L1 + L2 Hybrid):
   Combines L1 and L2 penalties: Loss + lambda_1 * sum(|beta_j|) + lambda_2 * sum(beta_j^2).
   Balances group feature selection with stable coefficient shrinkage.
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
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
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
    print("VITALSIGN: REGULARIZATION (RIDGE, LASSO, ELASTIC NET) (M2)")
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
        'number_emergency',
        'number_outpatient'
    ]

    data = df[predictor_cols + [target_col]].dropna().copy()
    X = data[predictor_cols]
    y = data[target_col]

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Standardize predictors (crucial for regularization penalties)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Models
    models = {
        'OLS Linear Regression': LinearRegression(),
        'Ridge Regression (L2)': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression (L1)': Lasso(alpha=0.01, random_state=42),
        'Elastic Net (L1+L2)': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    }

    results = []
    print("\n[1] Training and Evaluating Regularized Models on Holdout Test Set:")

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2_Score': round(r2, 4)
        })

    results_df = pd.DataFrame(results)
    print("\n[2] REGULARIZATION BENCHMARK SUMMARY:")
    print(results_df.to_string(index=False))

    # Save Comparison CSV
    out_csv = OUTPUT_DIR / "Regularization_Comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Comparison CSV saved to: {out_csv}")

    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: MAE & RMSE
    x = np.arange(len(results_df))
    width = 0.35
    axes[0].bar(x - width/2, results_df['MAE'], width, label="MAE (Days)", color="#3b82f6")
    axes[0].bar(x + width/2, results_df['RMSE'], width, label="RMSE (Days)", color="#ef4444")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results_df['Model'], rotation=15, ha='right', fontsize=9)
    axes[0].set_title("Error Comparison across Regularization Models", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Error (Days)", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: R2 Score
    axes[1].bar(results_df['Model'], results_df['R2_Score'], color="#10b981", width=0.45)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['Model'], rotation=15, ha='right', fontsize=9)
    axes[1].set_title("R2 Score Comparison", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("R-squared", fontsize=10)
    for i, v in enumerate(results_df['R2_Score']):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("VitalSign: Regularization (Ridge vs Lasso vs Elastic Net)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_png = OUTPUT_DIR / "Regularization_Comparison.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] Comparison plot saved to: {out_png}")
    print("=" * 70)

if __name__ == "__main__":
    main()
