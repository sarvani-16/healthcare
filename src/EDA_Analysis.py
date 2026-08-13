# ============================================================
# VITALSIGN - EXPLORATORY DATA ANALYSIS (EDA)
# Diabetes 130-US Hospitals Dataset
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = (
    'C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv'
)

OUTPUT_FOLDER = (
    'C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/VitalSign_EDA_Analysis_outputs'
)

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. PLOT STYLE
# ============================================================

sns.set(style="whitegrid")

plt.rcParams["figure.figsize"] = (8, 5)


# ============================================================
# 3. CORRECT COLUMN NAMES
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
# 4. LOAD DATASET
# ============================================================

print("=" * 70)
print("VITALSIGN - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\n1. Loading Dataset...")

try:

    # header=None because the uploaded CSV needs correct headers
    df = pd.read_csv(DATASET_PATH, header=None)

    # Assign correct column names
    df.columns = column_names

    print("\nDataset Loaded Successfully!")

except FileNotFoundError:

    print("\nERROR: Dataset file was not found.")
    print("Check the DATASET_PATH.")

    exit()

except Exception as e:

    print("\nERROR:", e)

    exit()


# ============================================================
# 5. DISPLAY FIRST FIVE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("2. First Five Records")
print("=" * 70)

print(df.head())


# ============================================================
# 6. DATASET SHAPE
# ============================================================

print("\n" + "=" * 70)
print("3. Dataset Shape")
print("=" * 70)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 7. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("4. Column Names")
print("=" * 70)

print(df.columns.tolist())


# ============================================================
# 8. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("5. Data Types")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 9. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("6. Dataset Information")
print("=" * 70)

df.info()


# ============================================================
# 10. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("7. Missing Values")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)


print("\nTotal Missing Values:")

total_missing = df.isnull().sum().sum()

print(total_missing)


# ============================================================
# 11. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("8. Duplicate Rows")
print("=" * 70)

duplicate_rows = df.duplicated().sum()

print("Duplicate Rows:", duplicate_rows)


# ============================================================
# 12. NUMERICAL COLUMNS
# ============================================================

numeric_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

print("\n" + "=" * 70)
print("9. Numerical Columns")
print("=" * 70)

print(numeric_cols)

print("\nNumber of Numerical Columns:")
print(len(numeric_cols))


# ============================================================
# 13. CATEGORICAL COLUMNS
# ============================================================

categorical_columns = df.select_dtypes(
    include=['object', 'category', 'bool']
).columns.tolist()

print("\n" + "=" * 70)
print("10. Categorical Columns")
print("=" * 70)

print(categorical_columns)

print("\nNumber of Categorical Columns:")
print(len(categorical_columns))


# ============================================================
# 14. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("11. Statistical Summary")
print("=" * 70)

summary = df.describe(include='all')

print(summary)

summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Statistical_Summary.csv"
    )
)


# ============================================================
# 15. UNIVARIATE ANALYSIS - HISTOGRAM
# ============================================================

print("\n" + "=" * 70)
print("12. Univariate Analysis - Histograms")
print("=" * 70)

for col in numeric_cols:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[col],
        bins=20,
        kde=True
    )

    plt.title(f"Histogram - {col}")

    plt.xlabel(col)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"Histogram_{col}.png"
        ),
        dpi=300
    )

    plt.close()


print("Histogram analysis completed.")


# ============================================================
# 16. BOX PLOTS
# ============================================================

print("\n" + "=" * 70)
print("13. Box Plot Analysis")
print("=" * 70)

for col in numeric_cols:

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        y=df[col]
    )

    plt.title(f"Box Plot - {col}")

    plt.ylabel(col)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"Boxplot_{col}.png"
        ),
        dpi=300
    )

    plt.close()


print("Box plot analysis completed.")


# ============================================================
# 17. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 70)
print("14. Outlier Detection")
print("=" * 70)

outlier_results = []

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    outlier_count = len(outliers)

    outlier_results.append({
        "Column": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outlier Count": outlier_count
    })


outlier_df = pd.DataFrame(outlier_results)

print(outlier_df)

outlier_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Outlier_Analysis.csv"
    ),
    index=False
)


# ============================================================
# 18. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("15. Correlation Matrix")
print("=" * 70)

numeric_df = df.select_dtypes(
    include=np.number
)

corr = numeric_df.corr()

print(corr)

corr.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Correlation_Matrix.csv"
    )
)


# ============================================================
# 19. CORRELATION HEATMAP
# ============================================================

print("\nGenerating Correlation Heatmap...")

plt.figure(figsize=(16, 12))

sns.heatmap(
    corr,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap - VitalSign Dataset"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Correlation_Heatmap.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 20. CATEGORICAL COUNT PLOTS
# ============================================================

print("\n" + "=" * 70)
print("16. Categorical Variable Analysis")
print("=" * 70)

for col in categorical_columns:

    # Avoid very high-cardinality columns
    if df[col].nunique() > 20:
        continue

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=col
    )

    plt.xticks(
        rotation=45,
        ha='right'
    )

    plt.title(
        f"Count Plot - {col}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"Countplot_{col}.png"
        ),
        dpi=300
    )

    plt.close()


print("Categorical analysis completed.")


# ============================================================
# 21. MISSING VALUE HEATMAP
# ============================================================

print("\nGenerating Missing Value Heatmap...")

plt.figure(figsize=(12, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title(
    "Missing Values Heatmap - VitalSign"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Missing_Values_Heatmap.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 22. TARGET VARIABLE
# ============================================================

target = "readmitted"

print("\n" + "=" * 70)
print("17. Target Variable Analysis")
print("=" * 70)

print("Target Variable:", target)

print("\nTarget Values:")

print(df[target].value_counts())


# ============================================================
# 23. TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=target
)

plt.title(
    "Readmission Distribution"
)

plt.xlabel(
    "Readmission Status"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Target_Distribution_Readmitted.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 24. TARGET PERCENTAGE
# ============================================================

print("\nReadmission Percentage:")

target_percentage = (
    df[target]
    .value_counts(normalize=True)
    * 100
)

print(target_percentage)

target_percentage.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Readmission_Percentage.csv"
    )
)


# ============================================================
# 25. LENGTH OF STAY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("18. Length of Stay Analysis")
print("=" * 70)

print(
    df["time_in_hospital"].describe()
)


plt.figure(figsize=(8, 5))

sns.histplot(
    df["time_in_hospital"],
    bins=14,
    kde=True
)

plt.title(
    "Distribution of Length of Hospital Stay"
)

plt.xlabel(
    "Time in Hospital (Days)"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Length_of_Stay_Distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 26. LENGTH OF STAY VS READMISSION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="readmitted",
    y="time_in_hospital"
)

plt.title(
    "Length of Stay vs Readmission"
)

plt.xlabel(
    "Readmission Status"
)

plt.ylabel(
    "Time in Hospital"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Length_of_Stay_vs_Readmission.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 27. AGE VS READMISSION
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="age",
    hue="readmitted"
)

plt.title(
    "Age Group vs Readmission"
)

plt.xlabel(
    "Age Group"
)

plt.ylabel(
    "Number of Patients"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Age_vs_Readmission.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 28. GENDER VS READMISSION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="readmitted"
)

plt.title(
    "Gender vs Readmission"
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Gender_vs_Readmission.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 29. LAB PROCEDURES VS READMISSION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="readmitted",
    y="num_lab_procedures"
)

plt.title(
    "Laboratory Procedures vs Readmission"
)

plt.xlabel(
    "Readmission Status"
)

plt.ylabel(
    "Number of Lab Procedures"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Lab_Procedures_vs_Readmission.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 30. MEDICATIONS VS READMISSION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="readmitted",
    y="num_medications"
)

plt.title(
    "Number of Medications vs Readmission"
)

plt.xlabel(
    "Readmission Status"
)

plt.ylabel(
    "Number of Medications"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Medications_vs_Readmission.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 31. PAIR PLOT
# ============================================================

print("\nGenerating Pair Plot...")

# Use selected important numerical columns.
# Using every numeric column can create an extremely
# large and slow pair plot.

pair_columns = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

pair_columns = [
    col for col in pair_columns
    if col in df.columns
]

if len(pair_columns) > 1:

    pair = sns.pairplot(
        df[pair_columns].sample(
            min(2000, len(df)),
            random_state=42
        )
    )

    pair.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Pairplot_Selected_Features.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 32. SCATTER PLOT
# ============================================================

x_col = "time_in_hospital"

y_col = "num_medications"


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x=x_col,
    y=y_col,
    hue="readmitted",
    s=50
)

plt.title(
    "Time in Hospital vs Number of Medications"
)

plt.xlabel(
    "Time in Hospital"
)

plt.ylabel(
    "Number of Medications"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Scatter_Time_vs_Medications.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 33. INSULIN VS READMISSION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="insulin",
    hue="readmitted"
)

plt.title(
    "Insulin Treatment vs Readmission"
)

plt.xlabel(
    "Insulin Treatment"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Insulin_vs_Readmission.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 34. DIABETES MEDICATION VS READMISSION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="diabetesMed",
    hue="readmitted"
)

plt.title(
    "Diabetes Medication vs Readmission"
)

plt.xlabel(
    "Diabetes Medication"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "DiabetesMedication_vs_Readmission.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 35. SAVE CLEAN EDA DATA
# ============================================================

print("\nSaving EDA Dataset...")

df.head(1000).to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "VitalSign_EDA_Sample.csv"
    ),
    index=False
)


# ============================================================
# 36. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("VITALSIGN EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Dataset Path:")
print(DATASET_PATH)

print("\nOutput Folder:")
print(OUTPUT_FOLDER)

print("\nTotal Records:")
print(df.shape[0])

print("\nTotal Features:")
print(df.shape[1])

print("\nNumerical Features:")
print(len(numeric_cols))

print("\nCategorical Features:")
print(len(categorical_columns))

print("\nTotal Missing Values:")
print(total_missing)

print("\nDuplicate Records:")
print(duplicate_rows)

print("\nTarget Variable:")
print(target)

print("\nTarget Distribution:")
print(df[target].value_counts())

print("\nAll EDA analysis files have been saved successfully.")

print("=" * 70)