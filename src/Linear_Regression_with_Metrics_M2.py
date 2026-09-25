"""
=============================================================================
VITALSIGN: LINEAR REGRESSION WITH METRICS (M2)
Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv)
Target: time_in_hospital (Continuous Length of Stay, in days)
Predictors: Clinical Numerical Attributes (Lab tests, procedures, meds, diagnoses)
=============================================================================

ML Concept - Ordinary Least Squares (OLS) Linear Regression:
Models the expected value of a continuous dependent variable Y as a linear
combination of independent explanatory features X:
    Y = beta_0 + beta_1 * X_1 + beta_2 * X_2 + ... + beta_p * X_p + epsilon

CRITICAL CLINICAL & SYLLABUS DISTINCTION:
Because patient readmission within 30 days is a binary classification problem
(Yes/No), Linear Regression is mathematically inappropriate for readmission
classification (which requires Logistic Regression). To satisfy the ML curriculum
requirement for continuous Linear Regression, we predict patient Length of Stay
('time_in_hospital') using clinical workload and diagnostic predictors.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
    print("VITALSIGN: LINEAR REGRESSION WITH EVALUATION METRICS (M2)")
    print("=" * 70)

    df = load_data()
    print(f"Dataset Loaded: {len(df):,} Rows, {df.shape[1]} Columns")

    # Target: Length of Stay (time_in_hospital)
    target_col = 'time_in_hospital'
    predictor_cols = [
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_diagnoses',
        'number_inpatient',
        'number_emergency'
    ]

    print(f"\n[1] Regression Target: '{target_col}' (Hospital Length of Stay in Days)")
    print(f"    Selected Clinical Predictors: {predictor_cols}")

    # Prepare X and y
    data = df[predictor_cols + [target_col]].dropna().copy()
    X = data[predictor_cols]
    y = data[target_col]

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\n[2] Data Split: Train = {len(X_train):,} encounters, Test = {len(X_test):,} encounters")

    # Fit Linear Regression Model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predictions
    y_pred = lr.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n[3] MODEL PERFORMANCE METRICS (Evaluation on Test Set):")
    print(f"    Mean Absolute Error (MAE):     {mae:.4f} days")
    print(f"    Mean Squared Error (MSE):      {mse:.4f}")
    print(f"    Root Mean Squared Error (RMSE): {rmse:.4f} days")
    print(f"    R-squared (R2 Score):          {r2:.4f}")

    print("\n[4] LEARNED REGRESSION COEFFICIENTS:")
    print(f"    Intercept (beta_0): {lr.intercept_:.4f}")
    coef_df = pd.DataFrame({
        'Feature': predictor_cols,
        'Coefficient': lr.coef_
    }).sort_values(by='Coefficient', ascending=False)
    print(coef_df.to_string(index=False))

    # Save metrics table
    metrics_df = pd.DataFrame([{
        'Model': 'Ordinary Least Squares Linear Regression',
        'Target': 'time_in_hospital',
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2_Score': round(r2, 4)
    }])
    metrics_path = OUTPUT_DIR / "Linear_Regression_Metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[OK] Metrics saved to: {metrics_path}")

    # Generate Evaluation Chart
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test[:200], y_pred[:200], alpha=0.5, color="#2563eb", edgecolors="k", s=30)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Ideal Fit (y = y_hat)")
    plt.title("Linear Regression: Actual vs Predicted Length of Stay (Days)", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Hospital Stay (Days)", fontsize=11)
    plt.ylabel("Predicted Hospital Stay (Days)", fontsize=11)
    plt.legend()
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "Linear_Regression_Actual_vs_Predicted.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Evaluation plot saved to: {plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()