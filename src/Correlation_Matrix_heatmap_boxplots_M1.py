import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Output directory
output_dir = (
    "C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/outputs/Boxplots_correlation/"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. DATASET PATH
# ============================================================

dataset_path = (
    "C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"
)


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

try:

    # header=None because we assign the correct column names
    df = pd.read_csv(
        dataset_path,
        header=None
    )

    df.columns = column_names

    print(
        "Dataset Loaded Successfully."
    )

    print(
        "Dataset Shape:",
        df.shape
    )

except FileNotFoundError:

    print(
        "ERROR: Dataset file not found."
    )

    print(
        "Check the dataset path:"
    )

    print(dataset_path)

    exit()

except Exception as e:

    print(
        "ERROR:",
        e
    )

    exit()


# ============================================================
# 5. DISPLAY BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)

print("VITALSIGN DATASET")

print("=" * 70)

print("\nFirst 5 Records:")

print(df.head())


# ============================================================
# 6. SELECT NUMERICAL COLUMNS
# ============================================================

numerical_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()


print("\n" + "=" * 70)

print("--- Numerical Columns ---")

print("=" * 70)

print(numerical_cols)

print(
    "\nNumber of Numerical Columns:",
    len(numerical_cols)
)


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)

print("--- Correlation Matrix ---")

print("=" * 70)

corr_matrix = df[
    numerical_cols
].corr()

print(corr_matrix)


# ============================================================
# 8. SAVE CORRELATION MATRIX
# ============================================================

correlation_csv_path = os.path.join(
    output_dir,
    "correlation_matrix.csv"
)

corr_matrix.to_csv(
    correlation_csv_path
)

print(
    "\nCorrelation matrix saved to:"
)

print(correlation_csv_path)


# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================

print("\nGenerating Correlation Heatmap...")


plt.figure(
    figsize=(14, 10)
)

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of Numerical Features - VitalSign",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()


heatmap_path = os.path.join(
    output_dir,
    "correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Exported heatmap to:"
)

print(heatmap_path)


# ============================================================
# 10. TARGET COLUMN
# ============================================================

target_col = "readmitted"


print("\n" + "=" * 70)

print("Target Column:")

print(target_col)

print("=" * 70)


# ============================================================
# 11. CHECK TARGET COLUMN
# ============================================================

if target_col in df.columns:

    print(
        "\nTarget column found successfully."
    )

    print(
        "\nTarget Distribution:"
    )

    print(
        df[target_col].value_counts()
    )

else:

    print(
        f"\nTarget column '{target_col}' "
        "not found in dataset."
    )


# ============================================================
# 12. BOX PLOTS
# ============================================================

if target_col in df.columns:

    print("\n" + "=" * 70)

    print(
        "Generating Boxplots:"
    )

    print("=" * 70)


    for col in numerical_cols:

        # Do not create a boxplot of target against itself
        if col == target_col:
            continue


        plt.figure(
            figsize=(7, 5)
        )


        sns.boxplot(
            data=df,
            x=target_col,
            y=col
        )


        plt.title(
            f"{col} vs {target_col}",
            fontsize=12,
            fontweight="bold"
        )


        plt.xlabel(
            "Readmission Status"
        )


        plt.ylabel(
            col
        )


        plt.tight_layout()


        boxplot_filename = (
            f"boxplot_{col}_vs_{target_col}.png"
        )


        boxplot_path = os.path.join(
            output_dir,
            boxplot_filename
        )


        plt.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        print(
            f"Exported boxplot: {boxplot_filename}"
        )


else:

    print(
        f"\nTarget column '{target_col}' "
        "not found. Skipping boxplots."
    )


# ============================================================
# 13. IMPORTANT FEATURES VS READMISSION
# ============================================================

important_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]


print("\n" + "=" * 70)

print(
    "Generating Important Feature Boxplots"
)

print("=" * 70)


for col in important_features:

    if (
        col in df.columns
        and target_col in df.columns
    ):

        plt.figure(
            figsize=(7, 5)
        )


        sns.boxplot(
            data=df,
            x=target_col,
            y=col
        )


        plt.title(
            f"{col} vs Readmission",
            fontsize=12,
            fontweight="bold"
        )


        plt.xlabel(
            "Readmission Status"
        )


        plt.ylabel(
            col
        )


        plt.tight_layout()


        filename = (
            f"Important_{col}_vs_Readmission.png"
        )


        path = os.path.join(
            output_dir,
            filename
        )


        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        print(
            f"Saved: {filename}"
        )


# ============================================================
# 14. TOP CORRELATIONS WITH TARGET
# ============================================================

if target_col in corr_matrix.columns:

    print("\n" + "=" * 70)

    print(
        "Correlation With Readmission"
    )

    print("=" * 70)


    target_correlations = (
        corr_matrix[target_col]
        .sort_values(
            ascending=False
        )
    )


    print(
        target_correlations
    )


    target_correlations.to_csv(
        os.path.join(
            output_dir,
            "readmission_correlations.csv"
        )
    )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print(
    "VITALSIGN CORRELATION & BOXPLOT ANALYSIS COMPLETED"
)

print("=" * 70)

print(
    "\nDataset:",
    dataset_path
)

print(
    "\nOutput Folder:",
    output_dir
)

print(
    "\nTarget:",
    target_col
)

print(
    "\nTotal Records:",
    len(df)
)

print(
    "\nTotal Numerical Features:",
    len(numerical_cols)
)

print(
    "\nAll correlation matrices and boxplots "
    "have been saved successfully."
)

print("=" * 70)