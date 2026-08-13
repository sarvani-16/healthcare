from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)
app.secret_key = "healthcare_prediction_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

df = None
model = None
scaler = None
label_encoder = None

model_accuracy = None
model_report = None


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset():

    path = r"C:/Users/Manepalli Sarvani/PycharmProjects/Healthcare/dataset/diabetic_data_50000.csv"

    if os.path.exists(path):
        return path

    return None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    global df

    path = find_dataset()

    if path is not None:

        try:
            df = pd.read_csv(path)
            return True

        except Exception as e:

            print("Dataset loading error:", e)
            return False

    return False


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html",
        dataset_loaded=df is not None
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template("about.html")


# ============================================================
# CONTACT
# ============================================================

@app.route("/contact")
def contact():

    return render_template("contact.html")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if df is None:
        load_dataset()

    if df is None:

        return render_template(
            "dashboard.html",
            total_records=0,
            total_features=0,
            missing_values=0,
            duplicate_records=0,
            model_accuracy=None
        )

    total_records = df.shape[0]

    total_features = df.shape[1]

    missing_values = int(df.isnull().sum().sum())

    duplicate_records = int(df.duplicated().sum())

    return render_template(
        "dashboard.html",
        total_records=total_records,
        total_features=total_features,
        missing_values=missing_values,
        duplicate_records=duplicate_records,
        model_accuracy=model_accuracy
    )


# ============================================================
# DATASET PAGE
# ============================================================

@app.route("/dataset")
def dataset():

    if df is None:
        load_dataset()

    if df is None:

        return render_template(
            "dataset.html",
            data=None,
            columns=[],
            total_records=0,
            total_features=0
        )

    data = df.head(100).to_html(
        classes="table table-striped table-bordered",
        index=False
    )

    return render_template(
        "dataset.html",
        data=data,
        columns=df.columns.tolist(),
        total_records=len(df),
        total_features=len(df.columns)
    )


# ============================================================
# VIEW DATASET
# ============================================================

@app.route("/view-dataset")
def view_dataset():

    if df is None:
        load_dataset()

    if df is None:

        flash("Please upload a healthcare dataset first.", "warning")

        return redirect(url_for("dataset"))

    data = df.head(100).to_html(
        classes="table table-striped table-bordered",
        index=False
    )

    return render_template(
        "view_dataset.html",
        data=data
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route("/upload-dataset", methods=["GET", "POST"])
def upload_dataset():

    global df

    if request.method == "POST":

        file = request.files.get("dataset")

        if file is None or file.filename == "":

            flash("Please select a CSV file.", "danger")

            return redirect(url_for("upload_dataset"))

        if not file.filename.lower().endswith(".csv"):

            flash("Only CSV files are supported.", "danger")

            return redirect(url_for("upload_dataset"))

        filepath = os.path.join(
            DATASET_FOLDER,
            file.filename
        )

        file.save(filepath)

        try:

            df = pd.read_csv(filepath)

            flash(
                "Healthcare dataset uploaded successfully!",
                "success"
            )

            return redirect(url_for("dashboard"))

        except Exception as e:

            flash(
                f"Error loading dataset: {e}",
                "danger"
            )

            return redirect(url_for("upload_dataset"))

    return render_template("dataset.html")


# ============================================================
# SUMMARY
# ============================================================

@app.route("/summary")
def summary():

    if df is None:
        load_dataset()

    if df is None:

        return render_template(
            "summary.html",
            summary=None,
            shape=None,
            missing=None
        )

    summary_data = df.describe(
        include="all"
    ).fillna("").to_html(
        classes="table table-bordered table-striped"
    )

    missing_data = df.isnull().sum().to_frame(
        "Missing Values"
    ).to_html(
        classes="table table-bordered table-striped"
    )

    return render_template(
        "summary.html",
        summary=summary_data,
        shape=df.shape,
        missing=missing_data
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing():

    if df is None:
        load_dataset()

    if df is None:

        return render_template(
            "preprocessing.html",
            message="No dataset loaded."
        )

    missing_before = int(
        df.isnull().sum().sum()
    )

    duplicate_before = int(
        df.duplicated().sum()
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    return render_template(
        "preprocessing.html",
        missing_before=missing_before,
        duplicate_before=duplicate_before,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns
    )


# ============================================================
# VISUALIZATION
# ============================================================

@app.route("/visualization")
def visualization():

    if df is None:
        load_dataset()

    if df is None:

        return render_template(
            "visualization.html"
        )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # ----------------------------------------
    # Missing values graph
    # ----------------------------------------

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) > 0:

        plt.figure(figsize=(10, 6))

        missing.plot(kind="bar")

        plt.title(
            "Missing Values by Column"
        )

        plt.xlabel("Column")

        plt.ylabel("Missing Values")

        plt.xticks(rotation=45)

        plt.tight_layout()

        missing_path = os.path.join(
            STATIC_FOLDER,
            "missing_values.png"
        )

        plt.savefig(
            missing_path,
            dpi=150
        )

        plt.close()

    # ----------------------------------------
    # Target distribution
    # ----------------------------------------

    target = None

    possible_targets = [
        "readmitted",
        "Readmitted",
        "readmission",
        "Readmission"
    ]

    for col in possible_targets:

        if col in df.columns:

            target = col
            break

    if target is not None:

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=df,
            x=target
        )

        plt.title(
            "Healthcare Readmission Distribution"
        )

        plt.xticks(rotation=30)

        plt.tight_layout()

        target_path = os.path.join(
            STATIC_FOLDER,
            "target_distribution.png"
        )

        plt.savefig(
            target_path,
            dpi=150
        )

        plt.close()

    # ----------------------------------------
    # Correlation heatmap
    # ----------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if len(numeric_df.columns) > 1:

        corr = numeric_df.corr()

        plt.figure(
            figsize=(12, 8)
        )

        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=False
        )

        plt.title(
            "Healthcare Dataset Correlation"
        )

        plt.tight_layout()

        corr_path = os.path.join(
            STATIC_FOLDER,
            "correlation_heatmap.png"
        )

        plt.savefig(
            corr_path,
            dpi=150
        )

        plt.close()

    return render_template(
        "visualization.html",
        numeric_columns=numeric_columns,
        target=target
    )


# ============================================================
# TRAIN MODEL
# ============================================================

@app.route("/models", methods=["GET", "POST"])
def models():

    global model
    global scaler
    global label_encoder
    global model_accuracy
    global model_report

    if df is None:
        load_dataset()

    if df is None:

        return render_template(
            "models.html",
            accuracy=None,
            report=None
        )

    if request.method == "POST":

        target = None

        possible_targets = [
            "readmitted",
            "Readmitted",
            "readmission",
            "Readmission"
        ]

        for col in possible_targets:

            if col in df.columns:

                target = col
                break

        if target is None:

            flash(
                "Target column 'readmitted' was not found.",
                "danger"
            )

            return redirect(
                url_for("models")
            )

        try:

            data = df.copy()

            # ------------------------------------
            # Remove unwanted columns
            # ------------------------------------

            columns_to_remove = [
                target,
                "encounter_id",
                "patient_nbr",
                "weight",
                "payer_code",
                "medical_specialty"
            ]

            columns_to_remove = [
                col for col in columns_to_remove
                if col in data.columns
            ]

            X = data.drop(
                columns=columns_to_remove
            )

            y = data[target]

            # ------------------------------------
            # Convert categorical columns
            # ------------------------------------

            X = pd.get_dummies(
                X,
                drop_first=True
            )

            # ------------------------------------
            # Replace missing values
            # ------------------------------------

            X = X.replace(
                "?",
                np.nan
            )

            X = X.apply(
                pd.to_numeric,
                errors="coerce"
            )

            X = X.fillna(
                X.median(numeric_only=True)
            )

            X = X.fillna(0)

            # ------------------------------------
            # Encode target
            # ------------------------------------

            label_encoder = LabelEncoder()

            y = label_encoder.fit_transform(
                y.astype(str)
            )

            # ------------------------------------
            # Train/Test Split
            # ------------------------------------

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            # ------------------------------------
            # Scaling
            # ------------------------------------

            scaler = StandardScaler()

            X_train = scaler.fit_transform(
                X_train
            )

            X_test = scaler.transform(
                X_test
            )

            # ------------------------------------
            # Random Forest
            # ------------------------------------

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train,
                y_train
            )

            # ------------------------------------
            # Prediction
            # ------------------------------------

            predictions = model.predict(
                X_test
            )

            model_accuracy = round(
                accuracy_score(
                    y_test,
                    predictions
                ) * 100,
                2
            )

            model_report = classification_report(
                y_test,
                predictions,
                zero_division=0
            )

            # ------------------------------------
            # Confusion Matrix
            # ------------------------------------

            cm = confusion_matrix(
                y_test,
                predictions
            )

            plt.figure(
                figsize=(7, 5)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues"
            )

            plt.title(
                "Healthcare Prediction Confusion Matrix"
            )

            plt.xlabel(
                "Predicted"
            )

            plt.ylabel(
                "Actual"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    STATIC_FOLDER,
                    "confusion_matrix.png"
                ),
                dpi=150
            )

            plt.close()

            flash(
                "Healthcare prediction model trained successfully!",
                "success"
            )

        except Exception as e:

            flash(
                f"Model training error: {e}",
                "danger"
            )

    return render_template(
        "models.html",
        accuracy=model_accuracy,
        report=model_report
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    global model

    result = None

    if request.method == "POST":

        if model is None:

            flash(
                "Please train the model before making a prediction.",
                "warning"
            )

            return redirect(
                url_for("models")
            )

        try:

            # ------------------------------------------------
            # Simple demonstration prediction
            # ------------------------------------------------
            #
            # For the complete patient prediction form,
            # we will connect every healthcare feature
            # to the trained model.
            #
            # ------------------------------------------------

            result = "Model is ready for healthcare prediction."

        except Exception as e:

            result = f"Prediction error: {e}"

    return render_template(
        "prediction.html",
        result=result
    )


# ============================================================
# REPORTS
# ============================================================

@app.route("/reports")
def reports():

    return render_template(
        "reports.html",
        accuracy=model_accuracy,
        report=model_report
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    load_dataset()

    print("=" * 60)
    print(" HEALTHCARE PREDICTION SYSTEM")
    print("=" * 60)

    if df is not None:

        print(
            "Dataset loaded successfully."
        )

        print(
            "Dataset Shape:",
            df.shape
        )

        print(
            "Columns:",
            len(df.columns)
        )

    else:

        print(
            "No healthcare dataset found."
        )

        print(
            "Put your CSV inside:"
        )

        print(
            DATASET_FOLDER
        )

    print("=" * 60)

    app.run(
        debug=True
    )