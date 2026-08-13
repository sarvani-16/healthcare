import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

file_path = (
    "C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"
)


# ============================================================
# 2. CORRECT COLUMN NAMES
# ============================================================

column_names = [
    'encounter_id',
    'patient_nbr',
    'race',
    'gender',
    'age',
    'weight',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'time_in_hospital',
    'payer_code',
    'medical_specialty',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'diag_1',
    'diag_2',
    'diag_3',
    'number_diagnoses',
    'max_glu_serum',
    'A1Cresult',
    'metformin',
    'repaglinide',
    'nateglinide',
    'chlorpropamide',
    'glimepiride',
    'acetohexamide',
    'glipizide',
    'glyburide',
    'tolbutamide',
    'pioglitazone',
    'rosiglitazone',
    'acarbose',
    'miglitol',
    'troglitazone',
    'tolazamide',
    'examide',
    'citoglipton',
    'insulin',
    'glyburide_metformin',
    'glipizide_metformin',
    'glimepiride_pioglitazone',
    'metformin_rosiglitazone',
    'metformin_pioglitazone',
    'change',
    'diabetesMed',
    'readmitted'
]


# ============================================================
# 3. READ DATASET
# ============================================================

# header=None because the uploaded CSV needs correct headers
df = pd.read_csv(
    file_path,
    header=None
)

# Assign correct column names
df.columns = column_names


# ============================================================
# 4. RETRIEVE DATA IN DIFFERENT WAYS
# ============================================================

print("=" * 60)
print("VITALSIGN DATASET")
print("=" * 60)


# ------------------------------------------------------------
# View first 5 rows
# ------------------------------------------------------------

print("\n--- First 5 Rows ---")

print(df.head())


# ------------------------------------------------------------
# View last 5 rows
# ------------------------------------------------------------

print("\n--- Last 5 Rows ---")

print(df.tail())


# ------------------------------------------------------------
# View first 10 rows
# ------------------------------------------------------------

print("\n--- First 10 Rows ---")

print(df.head(10))


# ============================================================
# 5. PRINT SPECIFIC COLUMNS
# ============================================================

print("\n--- Print First 6 Columns ---")

subset = df.iloc[:, 0:6]

print(subset)


# ============================================================
# 6. PRINT IMPORTANT VITALSIGN COLUMNS
# ============================================================

print("\n--- Important VitalSign Columns ---")

important_columns = [
    'gender',
    'age',
    'time_in_hospital',
    'num_lab_procedures',
    'num_medications',
    'number_diagnoses',
    'insulin',
    'diabetesMed',
    'readmitted'
]

print(
    df[important_columns].head(10)
)


# ============================================================
# 7. IDENTIFY MISSING VALUES PER COLUMN
# ============================================================

print("\n" + "=" * 60)

print("----- Missing Values Per Column -----")

missing_counts = df.isnull().sum()

print(missing_counts)

print("-" * 40)


# ============================================================
# 8. TOTAL MISSING VALUES
# ============================================================

total_missing = df.isnull().sum().sum()

print("\n----- Total Missing Values -----")

print(total_missing)


# ============================================================
# 9. COLUMNS HAVING MISSING VALUES
# ============================================================

print("\n----- Columns Having Missing Values -----")

missing_columns = missing_counts[
    missing_counts > 0
]

print(missing_columns)


# ============================================================
# 10. MISSING VALUE PERCENTAGE
# ============================================================

print("\n----- Missing Value Percentage -----")

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

print(
    missing_percentage[
        missing_percentage > 0
    ]
)


# ============================================================
# 11. DETECT DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)

print("----- Duplicate Records -----")

duplicate_rows = df[
    df.duplicated()
]

print(
    f"Total duplicate rows detected: "
    f"{len(duplicate_rows)}"
)

print(duplicate_rows)


# ============================================================
# 12. DISPLAY DATASET SHAPE
# ============================================================

print("\n" + "=" * 60)

print("----- Dataset Shape -----")

print(
    "Number of Rows    :",
    df.shape[0]
)

print(
    "Number of Columns :",
    df.shape[1]
)


# ============================================================
# 13. PRODUCE MISSINGNESS HEATMAP
# ============================================================

print("\nGenerating Missing Values Heatmap...")

plt.figure(
    figsize=(14, 6)
)

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title(
    "Missing Values Heatmap - VitalSign Dataset"
)

plt.xlabel(
    "Dataset Columns"
)

plt.ylabel(
    "Patient Records"
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. MISSING VALUES BAR CHART
# ============================================================

missing_for_plot = missing_counts[
    missing_counts > 0
]

if len(missing_for_plot) > 0:

    plt.figure(
        figsize=(12, 6)
    )

    missing_for_plot.sort_values(
        ascending=False
    ).plot(
        kind='bar'
    )

    plt.title(
        "Missing Values Per Column"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Number of Missing Values"
    )

    plt.xticks(
        rotation=45,
        ha='right'
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 15. READMISSION VALUES
# ============================================================

print("\n" + "=" * 60)

print("----- Readmission Values -----")

print(
    df['readmitted'].value_counts()
)


# ============================================================
# 16. READMISSION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
    x='readmitted'
)

plt.title(
    "Patient Readmission Distribution"
)

plt.xlabel(
    "Readmission Status"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.show()


# ============================================================
# 17. LENGTH OF HOSPITAL STAY
# ============================================================

print("\n" + "=" * 60)

print("----- Length of Hospital Stay -----")

print(
    df['time_in_hospital'].describe()
)


# ============================================================
# 18. LENGTH OF STAY HISTOGRAM
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    df['time_in_hospital'],
    bins=15,
    kde=True
)

plt.title(
    "Distribution of Hospital Stay"
)

plt.xlabel(
    "Time in Hospital (Days)"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)

print("VITALSIGN DATASET SUMMARY")

print("=" * 60)

print(
    "Total Records       :",
    df.shape[0]
)

print(
    "Total Features      :",
    df.shape[1]
)

print(
    "Total Missing Values:",
    total_missing
)

print(
    "Duplicate Records   :",
    len(duplicate_rows)
)

print(
    "Target Variable     :",
    "readmitted"
)

print("=" * 60)

print("\nData Inspection Completed Successfully.")