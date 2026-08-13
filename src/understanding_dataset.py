
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n1. Load the Dataset")
print("=" * 70)

file_path = 'C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv'

try:

    # --------------------------------------------------------
    # Correct column names for Diabetes 130-US Hospitals
    # --------------------------------------------------------

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

    # Read CSV
    # header=None because the uploaded file has incorrect/missing header
    df = pd.read_csv(file_path, header=None)

    # Assign correct column names
    df.columns = column_names

    # ========================================================
    # 2. DATASET CONTENTS
    # ========================================================

    print("-----------------------------------")
    print("1. Dataset Contents:")
    print("-----------------------------------")

    print(df)

    # ========================================================
    # 3. DATASET SHAPE
    # ========================================================

    print("-----------------------------------")
    print("\n2. Number of Rows and Columns:")
    print("-----------------------------------")

    print(df.shape)

    # ========================================================
    # 4. COLUMN NAMES
    # ========================================================

    print("-----------------------------------")
    print("\n3. Column Names:")
    print("-----------------------------------")

    print(df.columns.tolist())

    # Configure pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 100)

    # ========================================================
    # 5. FIRST 10 RECORDS
    # ========================================================

    print("-----------------------------------")
    print("\n4. --- VitalSign CSV Dataset Table View ---")
    print("-----------------------------------")

    print("Dataset First 10 Records")
    print("-----------------------------------")

    print(df.head(10))

    # ========================================================
    # 6. LAST 10 RECORDS
    # ========================================================

    print("****************************************************************")
    print("Dataset Last 10 Records")
    print("-----------------------------------")

    print(df.tail(10))

    # ========================================================
    # 7. UNDERSTAND DATASET
    # ========================================================

    print("-----------------------------------")
    print("2. Understand the Dataset")
    print("-----------------------------------")

    # ========================================================
    # 8. DATA TYPES
    # ========================================================

    print("-----------------------------------")
    print("\n5. Data Types of Columns:")
    print("-----------------------------------")

    print(df.dtypes)

    print("=" * 60)

    # ========================================================
    # 9. COLUMN NAME + DATA TYPE
    # ========================================================

    print("6. Display Column Names with Data Types")
    print("\nColumn Name\t\tData Type")
    print("-" * 50)

    for column in df.columns:
        print(f"{column:<30} {df[column].dtype}")

    # ========================================================
    # 10. DATASET INFORMATION
    # ========================================================

    print("-----------------------------------")
    print("7. Dataset Summary and Information")
    print("-----------------------------------")

    df.info()

    print("\n" + "=" * 60 + "\n")

    # ========================================================
    # 11. NUMERIC COLUMNS
    # ========================================================

    print("8. Display Numeric Columns")
    print("-----------------------------------")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    print("Numerical Columns:")
    print(numeric_df)

    # ========================================================
    # 12. MISSING VALUES IN NUMERIC COLUMNS
    # ========================================================

    print("\n9. Missing Values in Numeric Attributes")
    print("-----------------------------------")

    print(numeric_df.isnull().sum())

    print(
        "\n10. Total Missing Numeric Values:",
        numeric_df.isnull().sum().sum()
    )

    # ========================================================
    # 13. FLOAT COLUMNS
    # ========================================================

    float_columns = df.select_dtypes(
        include=['float64']
    ).columns

    print("-----------------------------------")
    print("11. Float Attribute Names:")
    print("-----------------------------------")

    for column in float_columns:
        print(column)

    # ========================================================
    # 14. MISSING VALUES IN FLOAT COLUMNS
    # ========================================================

    print("12. Missing Values in Float Attributes")
    print("=" * 50)

    print(
        df[float_columns].isnull().sum()
    )

    print(
        "\nTotal Missing Float Values:",
        df[float_columns].isnull().sum().sum()
    )

    # ========================================================
    # 15. CATEGORICAL COLUMNS
    # ========================================================

    categorical_df = df.select_dtypes(
        include=['object']
    )

    print("-----------------------------------")
    print("13. Display Categorical (Object) Attributes:")
    print("-----------------------------------")

    print(categorical_df)

    # ========================================================
    # 16. MISSING CATEGORICAL VALUES
    # ========================================================

    print("Missing Values in Categorical Attributes")
    print("=" * 50)

    print(categorical_df.isnull().sum())

    print(
        "\n14. Total Missing Categorical Values:",
        categorical_df.isnull().sum().sum()
    )

    # ========================================================
    # 17. MISSING VALUES IN EACH COLUMN
    # ========================================================

    print("-----------------------------------")
    print("15. Missing Values in Each Column")
    print("-" * 50)

    print(df.isnull().sum())

    # ========================================================
    # 18. TOTAL MISSING VALUES
    # ========================================================

    total_missing = df.isnull().sum().sum()

    print("-----------------------------------")
    print("16. Total Missing Values:")
    print(total_missing)

    # ========================================================
    # 19. DUPLICATE RECORDS
    # ========================================================

    print("-----------------------------------")

    duplicate_count = df.duplicated().sum()

    print("17. Number of Duplicate Records:")
    print(duplicate_count)

    # ========================================================
    # 20. STATISTICAL OVERVIEW
    # ========================================================

    print("-----------------------------------")
    print("18. Statistical Overview")
    print("-----------------------------------")

    print(df.describe())

    # ========================================================
    # 21. READMISSION DISTRIBUTION
    # ========================================================

    print("-----------------------------------")
    print("19. Readmission Distribution")
    print("-----------------------------------")

    print(df['readmitted'].value_counts())

    # ========================================================
    # 22. READMISSION PERCENTAGE
    # ========================================================

    print("-----------------------------------")
    print("20. Readmission Percentage")
    print("-----------------------------------")

    print(
        df['readmitted'].value_counts(normalize=True) * 100
    )

    # ========================================================
    # 23. HISTOGRAM - TIME IN HOSPITAL
    # ========================================================

    print("-----------------------------------")
    print("21. Display Histogram of Time in Hospital")
    print("-----------------------------------")

    plt.figure(figsize=(8, 5))

    plt.hist(
        df['time_in_hospital'],
        bins=10,
        edgecolor='black'
    )

    plt.title("Histogram of Time in Hospital")
    plt.xlabel("Time in Hospital (Days)")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()

    # ========================================================
    # 24. HISTOGRAM - NUMBER OF MEDICATIONS
    # ========================================================

    print("-----------------------------------")
    print("22. Display Histogram of Number of Medications")
    print("-----------------------------------")

    plt.figure(figsize=(8, 5))

    plt.hist(
        df['num_medications'],
        bins=15,
        edgecolor='black'
    )

    plt.title("Histogram of Number of Medications")
    plt.xlabel("Number of Medications")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()

    # ========================================================
    # 25. HISTOGRAM - LAB PROCEDURES
    # ========================================================

    print("-----------------------------------")
    print("23. Display Histogram of Lab Procedures")
    print("-----------------------------------")

    plt.figure(figsize=(8, 5))

    plt.hist(
        df['num_lab_procedures'],
        bins=20,
        edgecolor='black'
    )

    plt.title("Histogram of Laboratory Procedures")
    plt.xlabel("Number of Lab Procedures")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()

    # ========================================================
    # 26. READMISSION COUNT GRAPH
    # ========================================================

    print("-----------------------------------")
    print("24. Readmission Count Graph")
    print("-----------------------------------")

    plt.figure(figsize=(8, 5))

    df['readmitted'].value_counts().plot(
        kind='bar',
        edgecolor='black'
    )

    plt.title("Patient Readmission Distribution")
    plt.xlabel("Readmission Status")
    plt.ylabel("Number of Patients")

    plt.xticks(rotation=0)

    plt.grid(axis='y')

    plt.show()

    # ========================================================
    # 27. CORRELATION HEATMAP
    # ========================================================

    print("-----------------------------------")
    print("25. Correlation Heatmap")
    print("-----------------------------------")

    correlation = numeric_df.corr()

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        correlation,
        cmap='coolwarm',
        linewidths=0.5
    )

    plt.title("Correlation Heatmap - VitalSign Dataset")

    plt.show()

    # ========================================================
    # 28. DATASET SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VITALSIGN DATASET SUMMARY")
    print("=" * 70)

    print("Total Records       :", df.shape[0])
    print("Total Features      :", df.shape[1])
    print("Numeric Features    :", len(numeric_df.columns))
    print("Categorical Features:", len(categorical_df.columns))
    print("Missing Values      :", total_missing)
    print("Duplicate Records   :", duplicate_count)

    print("=" * 70)

except FileNotFoundError:

    print(
        f"Error: The file at '{file_path}' was not found."
    )

except Exception as e:

    print(f"An error occurred: {e}")