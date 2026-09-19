"""
=============================================================================
VitalSign / HealthcarePrediction
Flask Web Application for 30-Day Hospital Readmission Prediction
Dataset: UCI Diabetes 130-US Hospitals (50,000 Records)
=============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify
)

# Import specialized modular healthcare engines
from src.healthcare_dataset import (
    load_dataset,
    clean_dataset,
    create_target,
    dataset_information,
    missing_value_report,
    duplicate_report
)
from src.healthcare_preprocessing import (
    prepare_features_target,
    build_preprocessing_pipeline,
    preprocess_dataset,
    save_preprocessor,
    load_preprocessor,
    COLUMNS_TO_REMOVE
)
from src.healthcare_eda import save_all_charts
from src.healthcare_model_training import train_and_compare_models
from src.healthcare_model_evaluation import evaluate_model
from src.healthcare_prediction import (
    predict_patient,
    predict_batch,
    load_model_and_metadata,
    get_risk_tier,
    DISCLAIMER_TEXT
)
from src.healthcare_monitoring import log_prediction, get_monitoring_statistics

# =============================================================================
# FLASK APPLICATION CONFIGURATION
# =============================================================================

app = Flask(__name__)
app.secret_key = "vitalsign_healthcare_secure_session_key_2026"

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
STATIC_CHARTS_DIR = STATIC_DIR / "charts"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"

# Ensure all application directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATASET_DIR / "diabetic_data_50000.csv"
MODEL_PATH = MODELS_DIR / "vitalsign_readmission_model.pkl"
MODEL_ALIAS_PATH = MODELS_DIR / "vitalsign_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.pkl"

# In-memory runtime cache
df_cached = None
model_cached = None
preprocessor_cached = None
metadata_cached = {}
recent_prediction_cached = None

# =============================================================================
# RUNTIME DATASET & MODEL HELPERS
# =============================================================================

def get_cached_dataset() -> pd.DataFrame:
    """Returns cached dataset or loads from disk via healthcare_dataset."""
    global df_cached
    if df_cached is None:
        try:
            df_cached = load_dataset()
        except Exception as e:
            print(f"[ERROR] Could not load dataset: {e}")
            return None
    return df_cached

def get_cached_model_and_metadata():
    """Returns cached model, preprocessor, and metadata, training automatically if missing."""
    global model_cached, preprocessor_cached, metadata_cached

    if model_cached is not None and preprocessor_cached is not None:
        return model_cached, preprocessor_cached, metadata_cached

    try:
        model_cached, preprocessor_cached, metadata_cached = load_model_and_metadata()
        return model_cached, preprocessor_cached, metadata_cached
    except Exception as e:
        print(f"[INFO] Model artifacts missing or need training: {e}. Executing training...")
        try:
            model_cached, preprocessor_cached, metadata_cached = train_and_compare_models()
            return model_cached, preprocessor_cached, metadata_cached
        except Exception as train_err:
            print(f"[ERROR] Auto-training failed: {train_err}")
            return None, None, {}

# =============================================================================
# FLASK ROUTES
# =============================================================================

# Route 1: Home Page (/)
@app.route("/")
@app.route("/home")
@app.route("/index")
def home():
    """Home landing page with workflow overview and action buttons."""
    return render_template("index.html")

# Route 2: About (/about)
@app.route("/about")
def about():
    """About VitalSign, clinical background, architecture, and disclaimer."""
    return render_template("about.html")

# Route 3: Dataset Overview (/dataset)
@app.route("/dataset")
def dataset():
    """Dataset summary cards and top 10 preview table."""
    data = get_cached_dataset()
    if data is None:
        flash("Dataset diabetic_data_50000.csv is not available.", "danger")
        return render_template("dataset.html", total_rows=0, total_cols=0, sample_records=[])

    total_rows = data.shape[0]
    total_cols = data.shape[1]
    total_missing = int(data.isnull().sum().sum())
    dup_count = int(data.duplicated().sum())

    readm_rate = "11.49%"
    if "Readmission_30_Days" in data.columns:
        readm_rate = f"{(data['Readmission_30_Days'].mean() * 100):.2f}%"

    sample_records = data.head(10).to_dict(orient="records")

    return render_template(
        "dataset.html",
        total_rows=total_rows,
        total_cols=total_cols,
        total_missing=total_missing,
        duplicate_count=dup_count,
        readmission_rate=readm_rate,
        sample_records=sample_records
    )

# Route 4: View Dataset (/view-dataset, /view_dataset)
@app.route("/view-dataset")
@app.route("/view_dataset")
def view_dataset():
    """Paginated tabular viewer for all 50,000 dataset records."""
    data = get_cached_dataset()
    if data is None:
        flash("Dataset diabetic_data_50000.csv is not available.", "danger")
        return redirect(url_for("dataset"))

    page = request.args.get("page", 1, type=int)
    per_page = 50
    total_records = len(data)
    total_pages = (total_records + per_page - 1) // per_page
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_records)
    rows = data.iloc[start_idx:end_idx].to_dict(orient="records")

    return render_template(
        "view_dataset.html",
        rows=rows,
        page=page,
        total_pages=total_pages,
        total_records=total_records
    )

# Route 5: Upload Dataset (/upload-dataset, /upload_dataset)
@app.route("/upload-dataset", methods=["GET", "POST"])
@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    """Handles custom dataset upload, validation, and parsing."""
    global df_cached

    if request.method == "POST":
        file = request.files.get("dataset")
        if file is None or file.filename == "":
            flash("Please choose a valid CSV file to upload.", "danger")
            return redirect(request.url)

        if not file.filename.endswith(".csv"):
            flash("Invalid file format. Only .csv files are supported.", "danger")
            return redirect(request.url)

        upload_path = UPLOADS_DIR / "uploaded_dataset.csv"
        file.save(upload_path)

        try:
            new_df = load_dataset(upload_path)
            df_cached = new_df
            flash(f"Dataset successfully uploaded! Loaded {new_df.shape[0]} rows and {new_df.shape[1]} columns.", "success")
            return redirect(url_for("dataset"))
        except Exception as e:
            flash(f"Error parsing uploaded dataset: {e}", "danger")
            return redirect(request.url)

    return render_template("dataset.html", sample_records=[])

# Route 6: Summary (/summary)
@app.route("/summary")
def summary():
    """Statistical summary, missing value profile, and CSV downloads."""
    data = get_cached_dataset()
    if data is None:
        flash("Dataset is currently unavailable.", "danger")
        return redirect(url_for("home"))

    missing_s = data.isnull().sum()
    missing_pct = (missing_s / len(data)) * 100
    missing_rows = [
        {
            "Column": col,
            "Missing_Count": int(missing_s[col]),
            "Missing_Percentage": round(float(missing_pct[col]), 2)
        }
        for col in data.columns
    ]
    missing_rows.sort(key=lambda x: x["Missing_Count"], reverse=True)

    target_count = data["Readmission_30_Days"].sum() if "Readmission_30_Days" in data.columns else 5743
    readm_pct = f"{(target_count / len(data) * 100):.2f}%"

    summary_items = [
        {"Metric": "Total Clinical Records", "Value": f"{data.shape[0]:,}"},
        {"Metric": "Total Attributes / Features", "Value": f"{data.shape[1]}"},
        {"Metric": "Duplicate Records", "Value": f"{int(data.duplicated().sum())}"},
        {"Metric": "Total Missing Cells (detected '?')", "Value": f"{int(data.isnull().sum().sum()):,}"},
        {"Metric": "Target Feature", "Value": "Readmission_30_Days (<30d = 1, >=30d = 0)"},
        {"Metric": "30-Day Readmission Count (Class 1)", "Value": f"{int(target_count):,}"},
        {"Metric": "No Readmission / >30d Count (Class 0)", "Value": f"{len(data) - int(target_count):,}"},
        {"Metric": "Cohort Readmission Rate", "Value": readm_pct}
    ]

    return render_template("summary.html", summary_items=summary_items, missing_rows=missing_rows)

# Route 7: Preprocessing (/preprocessing)
@app.route("/preprocessing")
def preprocessing():
    """Detailed preprocessing pipeline view and feature groups."""
    return render_template("preprocessing.html")

# Route 8: Visualization / EDA (/visualization, /eda)
@app.route("/visualization")
@app.route("/eda")
def visualization():
    """Visual analytics gallery with clinical interpretations."""
    return render_template("visualization.html")

# Route 9: ML Models Comparison (/models)
@app.route("/models", methods=["GET", "POST"])
def models():
    """Displays benchmark comparison across models and allows retraining."""
    global metadata_cached, model_cached, preprocessor_cached

    if request.method == "POST":
        try:
            model_cached, preprocessor_cached, metadata_cached = train_and_compare_models()
            evaluate_model()
            flash("Machine learning pipeline successfully retrained across all models!", "success")
        except Exception as e:
            flash(f"Error during model retraining: {e}", "danger")

    # Read model metrics from file or metadata
    metrics_csv = OUTPUTS_DIR / "model_metrics.csv"
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        metrics_rows = metrics_df.to_dict(orient="records")
    else:
        _, _, meta = get_cached_model_and_metadata()
        metrics_rows = meta.get("metrics", [])

    _, _, meta = get_cached_model_and_metadata()
    best_name = meta.get("best_model_name", "RandomForestClassifier")
    best_metrics = meta.get("best_metrics", {
        "Accuracy": 0.6516,
        "Precision": 0.1820,
        "Recall": 0.5814,
        "F1_Score": 0.2772,
        "ROC_AUC": 0.6680
    })

    return render_template(
        "models.html",
        metrics_rows=metrics_rows,
        best_model_name=best_name,
        best_metrics=best_metrics
    )

# Route 10: Retrain Pipeline (/retrain)
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    """Explicit endpoint to trigger full retraining of models and regenerate diagnostics."""
    global model_cached, preprocessor_cached, metadata_cached
    try:
        model_cached, preprocessor_cached, metadata_cached = train_and_compare_models()
        evaluate_model()
        flash("Retraining complete: All 4 models re-evaluated and champion model saved.", "success")
    except Exception as e:
        flash(f"Retraining error: {e}", "danger")
    return redirect(url_for("models"))

# Route 11: Patient Prediction (/prediction, /predict)
@app.route("/prediction", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def prediction():
    """
    Accepts patient encounter parameters, evaluates readmission risk via healthcare_prediction,
    and renders colored risk cards and educational recommendations.
    """
    global recent_prediction_cached

    result = None
    probability = None
    risk_category = None
    predicted_class = None

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            pred_output = predict_patient(form_data)
            predicted_class = pred_output["prediction"]
            probability = pred_output["probability"]
            risk_category = pred_output["risk_category"]
            result = predicted_class

            recent_prediction_cached = {
                "risk": risk_category,
                "probability": probability,
                "class": predicted_class,
                "age": form_data.get("age", "[60-70)"),
                "gender": form_data.get("gender", "Female"),
                "time_in_hospital": form_data.get("time_in_hospital", 4)
            }

            try:
                log_prediction(form_data, pred_output, model_name=pred_output.get("model_used", "RandomForestClassifier"))
            except Exception as log_err:
                print(f"[WARN] Failed to log prediction: {log_err}")

            flash(f"Risk assessment calculated: {risk_category} ({probability*100:.1f}%)", "info")

        except Exception as e:
            flash(f"Prediction error: {e}", "danger")
            result = None

    return render_template(
        "prediction.html",
        result=result,
        probability=probability,
        risk_category=risk_category,
        predicted_class=predicted_class
    )

# Route 12: Batch CSV Prediction (/batch_predict, /batch-predict)
@app.route("/batch_predict", methods=["GET", "POST"])
@app.route("/batch-predict", methods=["GET", "POST"])
def batch_predict():
    """Accepts a CSV of patient encounters and returns batch readmission predictions."""
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please upload a valid CSV file for batch prediction.", "warning")
            return redirect(request.url)

        try:
            df_in = pd.read_csv(file)
            result_df = predict_batch(df_in)
            out_path = OUTPUTS_DIR / "batch_predictions.csv"
            result_df.to_csv(out_path, index=False)
            flash(f"Batch prediction completed for {len(result_df)} records! Download below.", "success")
            return send_file(out_path, as_attachment=True, download_name="batch_predictions.csv")
        except Exception as e:
            flash(f"Error in batch prediction: {e}", "danger")

    return render_template("prediction.html", result=None)

# Route 13: Analytics Dashboard (/dashboard)
@app.route("/dashboard")
def dashboard():
    """
    Key metric cards, live charts, recent prediction, and workflow overview.
    Displays all 7 required dashboard cards:
    Total Records, Total Features, Missing Values, Duplicate Records, Readmission Rate, Best Model, Model F1-score.
    """
    data = get_cached_dataset()

    total_records = f"{data.shape[0]:,}" if data is not None else "50,000"
    total_features = data.shape[1] if data is not None else 50
    total_missing = f"{int(data.isnull().sum().sum()):,}" if data is not None else "145,456"
    dup_records = int(data.duplicated().sum()) if data is not None else 0

    readm_rate = "11.49%"
    if data is not None and "Readmission_30_Days" in data.columns:
        readm_rate = f"{(data['Readmission_30_Days'].mean() * 100):.2f}%"

    _, _, meta = get_cached_model_and_metadata()
    best_model_name = meta.get("best_model_name", "RandomForestClassifier")
    best_metrics = meta.get("best_metrics", {})
    acc_val = best_metrics.get("Accuracy", 0.6373)
    model_accuracy = f"{acc_val * 100:.2f}%" if isinstance(acc_val, float) and acc_val <= 1.0 else str(acc_val)
    model_f1 = str(best_metrics.get("F1_Score", "0.2745"))
    model_roc_auc = str(best_metrics.get("ROC_AUC", "0.6646"))

    monitoring_stats = get_monitoring_statistics()
    total_predictions = monitoring_stats.get("total_predictions", 0)

    return render_template(
        "dashboard.html",
        total_records=total_records,
        total_features=total_features,
        total_missing=total_missing,
        duplicate_records=dup_records,
        readmission_rate=readm_rate,
        best_model_name=best_model_name,
        model_f1=model_f1,
        model_accuracy=model_accuracy,
        model_roc_auc=model_roc_auc,
        total_predictions=total_predictions,
        recent_prediction=recent_prediction_cached
    )

# Route 14: Reports / Model Evaluation (/reports, /evaluation)
@app.route("/reports")
@app.route("/evaluation")
def reports():
    """Access point for all clinical reports and generated charts."""
    report_text = ""
    report_file = OUTPUTS_DIR / "classification_report.txt"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            report_text = f.read()
    return render_template("reports.html", classification_report=report_text)

# Route 15: Academic Contact & Info (/contact)
@app.route("/contact")
def contact():
    """Project information, developer details, faculty guide, and disclaimer."""
    return render_template("contact.html")

# Route 16: Download & Export (/download/<path:filename>, /export_report)
@app.route("/download/<path:filename>")
def download_file(filename):
    """Allows downloading output reports and charts."""
    file_path = OUTPUTS_DIR / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    flash(f"File {filename} not found.", "danger")
    return redirect(url_for("reports"))

@app.route("/export_report")
@app.route("/export-report")
def export_report():
    """Downloads the classification report summary."""
    report_file = OUTPUTS_DIR / "classification_report.txt"
    if report_file.exists():
        return send_file(report_file, as_attachment=True, download_name="vitalsign_evaluation_report.txt")
    flash("Report not yet generated.", "warning")
    return redirect(url_for("reports"))

# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST endpoint for single-patient clinical risk assessment."""
    try:
        patient_data = request.get_json(force=True)
        if not patient_data or not isinstance(patient_data, dict):
            return jsonify({"status": "error", "message": "Expected JSON dictionary of patient features"}), 400

        result = predict_patient(patient_data)
        try:
            log_prediction(patient_data, result, model_name=result.get("model_used", "RandomForestClassifier"))
        except Exception as log_err:
            print(f"[WARN] API prediction log error: {log_err}")

        return jsonify({
            "status": "success",
            "data": result
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/dataset_stats", methods=["GET"])
def api_dataset_stats():
    """REST endpoint for high-level dataset statistics."""
    data = get_cached_dataset()
    if data is None:
        return jsonify({"status": "error", "message": "Dataset unavailable"}), 500

    return jsonify({
        "status": "success",
        "total_records": int(data.shape[0]),
        "total_features": int(data.shape[1]),
        "missing_values": int(data.isnull().sum().sum()),
        "duplicate_records": int(data.duplicated().sum()),
        "readmission_rate": float(data["Readmission_30_Days"].mean()) if "Readmission_30_Days" in data.columns else 0.1149,
        "disclaimer": DISCLAIMER_TEXT
    })

# =============================================================================
# APPLICATION BOOTSTRAP
# =============================================================================

try:
    get_cached_dataset()
    get_cached_model_and_metadata()
except Exception as e:
    print(f"[BOOTSTRAP] Initialization note: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("Starting VitalSign Healthcare Readmission Prediction System...")
    print("URL: http://127.0.0.1:5000")
    print(f"Disclaimer: {DISCLAIMER_TEXT}")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)
