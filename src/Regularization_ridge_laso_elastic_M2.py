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
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
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
print("VITALSIGN - REGULARIZATION: RIDGE, LASSO & ELASTIC NET (M2)")
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
    'number_emergency',
    'number_outpatient'
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
# 6. MODEL / ANALYSIS (RIDGE, LASSO, ELASTIC NET)
# ============================================================
# L1 Penalty: Lasso (drives coefficients to 0 for feature selection)
# L2 Penalty: Ridge (shrinks coefficients towards 0 to reduce variance)
# L1 + L2 Penalty: Elastic Net (hybrid balance)
models = {
    'OLS Linear Regression': LinearRegression(),
    'Ridge (L2 Penalty)': Ridge(alpha=1.0, random_state=42),
    'Lasso (L1 Penalty)': Lasso(alpha=0.01, random_state=42),
    'Elastic Net (L1+L2)': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    results.append({
        'Model': name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2_Score': round(r2, 4)
    })

# ============================================================
# 7. EVALUATION / SUMMARY
# ============================================================
reg_df = pd.DataFrame(results)
print("\n--- REGULARIZATION PERFORMANCE COMPARISON ---")
print(reg_df.to_string(index=False))

# ============================================================
# 8. SAVE RESULTS
# ============================================================
csv_path = os.path.join(OUTPUT_FOLDER, "regularization_comparison.csv")
reg_df.to_csv(csv_path, index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(reg_df))
width = 0.35

axes[0].bar(x - width/2, reg_df['MAE'], width, label='MAE (Days)', color='#3b82f6')
axes[0].bar(x + width/2, reg_df['RMSE'], width, label='RMSE (Days)', color='#ef4444')
axes[0].set_xticks(x)
axes[0].set_xticklabels(reg_df['Model'], rotation=15, ha='right', fontsize=9)
axes[0].set_ylabel('Error in Days', fontsize=10)
axes[0].set_title('Error Comparison across Regularization Penalties', fontweight='bold', fontsize=11)
axes[0].legend()

axes[1].bar(reg_df['Model'], reg_df['R2_Score'], color='#10b981', width=0.45)
axes[1].set_xticks(x)
axes[1].set_xticklabels(reg_df['Model'], rotation=15, ha='right', fontsize=9)
axes[1].set_ylabel('R-squared Score', fontsize=10)
axes[1].set_title('Variance Explained (R2 Score)', fontweight='bold', fontsize=11)
for i, v in enumerate(reg_df['R2_Score']):
    axes[1].text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9)

plt.suptitle('VitalSign: Ridge vs Lasso vs Elastic Net Regularization', fontsize=13, fontweight='bold')
plt.tight_layout()

chart_path = os.path.join(OUTPUT_FOLDER, "regularization_comparison.png")
plt.savefig(chart_path, dpi=300)
plt.close()

print(f"\n[OK] Comparison CSV saved to:   {csv_path}")
print(f"[OK] Comparison chart saved to: {chart_path}")

# ============================================================
# 9. FINAL OUTPUT
# ============================================================
print("=" * 60)
print("VITALSIGN: REGULARIZATION ANALYSIS COMPLETED")
print("=" * 60)
