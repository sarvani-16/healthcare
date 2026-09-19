# --------------------------------------------
# Placement Prediction Dataset Preprocessing
# --------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# --------------------------------------------
# Read original dataset
# --------------------------------------------

input_file = "C:/Users/Manepalli Sarvani/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv"

output_file = "C:/Users/Manepalli Sarvani/PycharmProjects/placement_prediction/dataset/final_preprocess_M2.csv"


df = pd.read_csv(input_file)


# --------------------------------------------
# Create copy
# --------------------------------------------

processed_df = df.copy()


print("=" * 60)
print("PLACEMENT PREDICTION DATASET PREPROCESSING")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(processed_df.shape)


# ============================================================
# 1. REMOVE DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("1. DUPLICATE RECORD REMOVAL")
print("=" * 60)

duplicates_before = processed_df.duplicated().sum()

print("Duplicate Records:", duplicates_before)


processed_df = processed_df.drop_duplicates().copy()


print(
    "Rows after duplicate removal:",
    len(processed_df)
)


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("2. MISSING VALUE HANDLING")
print("=" * 60)


# --------------------------------------------
# Numeric columns
# --------------------------------------------

numeric_cols = processed_df.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()


print("\nNumerical Columns:")
print(numeric_cols)


for col in numeric_cols:

    missing_count = processed_df[col].isnull().sum()

    if missing_count > 0:

        median_value = processed_df[col].median()

        processed_df[col] = processed_df[col].fillna(
            median_value
        )

        print(
            col,
            "->",
            missing_count,
            "missing values replaced with median:",
            median_value
        )


# --------------------------------------------
# Categorical columns
# --------------------------------------------

categorical_cols = processed_df.select_dtypes(
    include=['object', 'string']
).columns.tolist()


print("\nCategorical Columns:")
print(categorical_cols)


for col in categorical_cols:

    missing_count = processed_df[col].isnull().sum()

    if missing_count > 0:

        mode_values = processed_df[col].mode()

        if len(mode_values) > 0:

            mode_value = mode_values.iloc[0]

            processed_df[col] = processed_df[col].fillna(
                mode_value
            )

            print(
                col,
                "->",
                missing_count,
                "missing values replaced with mode:",
                mode_value
            )


# ============================================================
# 3. CLEAN TEXT DATA
# ============================================================

print("\n" + "=" * 60)
print("3. TEXT DATA CLEANING")
print("=" * 60)


for col in categorical_cols:

    # Remove leading/trailing spaces
    processed_df[col] = (
        processed_df[col]
        .astype(str)
        .str.strip()
    )

    # Convert to lowercase
    processed_df[col] = (
        processed_df[col]
        .str.lower()
    )


print("Text cleaning completed.")


# ============================================================
# 4. LABEL ENCODING
# ============================================================

print("\n" + "=" * 60)
print("4. LABEL ENCODING")
print("=" * 60)


for col in categorical_cols:

    encoder = LabelEncoder()

    processed_df[col] = encoder.fit_transform(
        processed_df[col]
    )

    print(
        col,
        "-> Label encoded"
    )


# ============================================================
# 5. FEATURE SCALING
# ============================================================

print("\n" + "=" * 60)
print("5. FEATURE SCALING")
print("=" * 60)


# ------------------------------------------------------------
# IMPORTANT:
#
# Do NOT scale:
#
# StudentID       -> identifier
# Salary Package  -> actual salary value
# PlacementStatus -> target 0/1
# IsAnomaly       -> 0/1 indicator
#
# Scale only continuous academic/performance features.
# ------------------------------------------------------------

scale_columns = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular"
]


# Check which columns actually exist

scale_columns = [
    col
    for col in scale_columns
    if col in processed_df.columns
]


print("\nColumns selected for scaling:")

print(scale_columns)


# ------------------------------------------------------------
# Apply StandardScaler
# ------------------------------------------------------------

scaler = StandardScaler()


processed_df[scale_columns] = scaler.fit_transform(
    processed_df[scale_columns]
)


print("\nFeature scaling completed.")


# ============================================================
# 6. VERIFY IMPORTANT COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("6. IMPORTANT COLUMN CHECK")
print("=" * 60)


# These should NOT be scaled

if "StudentID" in processed_df.columns:

    print(
        "\nStudentID sample:"
    )

    print(
        processed_df["StudentID"].head()
    )


if "Salary Package" in processed_df.columns:

    print(
        "\nSalary Package sample:"
    )

    print(
        processed_df["Salary Package"].head()
    )


if "PlacementStatus" in processed_df.columns:

    print(
        "\nPlacementStatus values:"
    )

    print(
        processed_df["PlacementStatus"].unique()
    )


if "IsAnomaly" in processed_df.columns:

    print(
        "\nIsAnomaly values:"
    )

    print(
        processed_df["IsAnomaly"].unique()
    )


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("7. MISSING VALUE CHECK")
print("=" * 60)


missing_values = processed_df.isnull().sum()


print(missing_values)


print(
    "\nTotal remaining missing values:",
    missing_values.sum()
)


# ============================================================
# 8. FINAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("FINAL PREPROCESSED DATASET")
print("=" * 60)


print("\nFirst 5 rows:")

print(
    processed_df.head()
)


print("\nFinal Dataset Shape:")

print(
    processed_df.shape
)


# ============================================================
# 9. SAVE DATASET
# ============================================================

processed_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)


print(
    "\nOriginal Dataset Shape :",
    df.shape
)


print(
    "Processed Dataset Shape:",
    processed_df.shape
)


print(
    "\nSaved File:"
)


print(
    output_file
)


print(
    "\nOriginal dataset was NOT modified."
)