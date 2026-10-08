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
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
print("VITALSIGN - NORMAL EQUATION (CFNE) vs GRADIENT DESCENT (GD) (M2)")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if "readmitted" in first_line or "encounter_id" in first_line:
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)

data = df.copy()

# ============================================================
# 5. DATA PREPROCESSING (CONTINUOUS LENGTH OF STAY)
# ============================================================
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. MODEL / ANALYSIS (CFNE vs SGD)
# ============================================================
# 1. Closed-Form Normal Equation (OLS)
cfne_model = LinearRegression()
cfne_model.fit(X_train_scaled, y_train)
y_pred_cfne = cfne_model.predict(X_test_scaled)

# 2. Gradient Descent (Iterative SGD)
gd_model = SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3, random_state=42)
gd_model.fit(X_train_scaled, y_train)
y_pred_gd = gd_model.predict(X_test_scaled)

# ============================================================
# 7. EVALUATION / COMPARISON
# ============================================================
comparison_results = [
    {
        'Optimization_Method': 'Closed-Form Normal Equation (OLS)',
        'MAE': round(mean_absolute_error(y_test, y_pred_cfne), 4),
        'MSE': round(mean_squared_error(y_test, y_pred_cfne), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_cfne)), 4),
        'R2_Score': round(r2_score(y_test, y_pred_cfne), 4)
    },
    {
        'Optimization_Method': 'Stochastic Gradient Descent (SGD)',
        'MAE': round(mean_absolute_error(y_test, y_pred_gd), 4),
        'MSE': round(mean_squared_error(y_test, y_pred_gd), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_gd)), 4),
        'R2_Score': round(r2_score(y_test, y_pred_gd), 4)
    }
]

comparison_df = pd.DataFrame(comparison_results)
print("\n--- OPTIMIZATION METHOD BENCHMARK ---")
print(comparison_df.to_string(index=False))

# ============================================================
# 8. SAVE RESULTS
# ============================================================
csv_path = os.path.join(OUTPUT_FOLDER, "Linear_Regression_CFNE_GD_Comparison.csv")
comparison_df.to_csv(csv_path, index=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(comparison_df))
width = 0.35
ax.bar(x - width/2, comparison_df['MAE'], width, label='MAE (Days)', color='#3b82f6')
ax.bar(x + width/2, comparison_df['RMSE'], width, label='RMSE (Days)', color='#10b981')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Optimization_Method'], fontsize=10)
ax.set_ylabel('Error Metric in Days', fontsize=11)
ax.set_title('Normal Equation vs Gradient Descent: Error Comparison', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()

chart_path = os.path.join(OUTPUT_FOLDER, "CFNE_vs_GD_Comparison.png")
plt.savefig(chart_path, dpi=300)
plt.close()

print(f"\n[OK] Comparison CSV saved to:   {csv_path}")
print(f"[OK] Comparison chart saved to: {chart_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: CFNE vs GD COMPARISON COMPLETED")
print("=" * 60)
