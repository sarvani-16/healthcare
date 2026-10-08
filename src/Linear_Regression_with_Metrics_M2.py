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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
print("VITALSIGN - LINEAR REGRESSION WITH EVALUATION METRICS (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (CONTINUOUS REGRESSION TARGET)
# ============================================================
# ACADEMIC NOTE:
# The primary project goal is binary readmission classification.
# Because Linear Regression is mathematically inappropriate for binary outcomes,
# we predict the patient's continuous hospital stay ('time_in_hospital' in days).

target_col = 'time_in_hospital'
predictor_cols = [
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_diagnoses',
    'number_inpatient',
    'number_emergency'
]

reg_data = data[predictor_cols + [target_col]].dropna().copy()
X = reg_data[predictor_cols]
y = reg_data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ============================================================
# 6. MODEL / ANALYSIS (FIT OLS LINEAR REGRESSION)
# ============================================================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# ============================================================
# 7. EVALUATION / METRICS CALCULATION
# ============================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Target Feature:               '{target_col}' (Length of Stay in Days)")
print(f"Mean Absolute Error (MAE):     {mae:.4f} days")
print(f"Mean Squared Error (MSE):      {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} days")
print(f"R-squared Score (R2):          {r2:.4f}")

metrics_df = pd.DataFrame([{
    'Model': 'Linear Regression (OLS)',
    'Target_Feature': 'time_in_hospital',
    'MAE': round(mae, 4),
    'MSE': round(mse, 4),
    'RMSE': round(rmse, 4),
    'R2_Score': round(r2, 4)
}])

# ============================================================
# 8. SAVE RESULTS
# ============================================================
csv_path = os.path.join(OUTPUT_FOLDER, "linear_regression_metrics.csv")
metrics_df.to_csv(csv_path, index=False)

plt.figure(figsize=(7, 6))
plt.scatter(y_test[:200], y_pred[:200], alpha=0.5, color="#2563eb", edgecolors="k", s=30)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Ideal Fit (y = y_hat)")
plt.title("Linear Regression: Actual vs Predicted Stay (Days)", fontsize=13, fontweight="bold")
plt.xlabel("Actual Hospital Stay (Days)", fontsize=11)
plt.ylabel("Predicted Hospital Stay (Days)", fontsize=11)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(OUTPUT_FOLDER, "Linear_Regression_Actual_vs_Predicted.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\n[OK] Linear regression metrics saved to: {csv_path}")
print(f"[OK] Evaluation plot saved to:            {plot_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: LINEAR REGRESSION COMPLETED")
print("=" * 60)