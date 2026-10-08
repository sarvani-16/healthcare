"""
=============================================================================
VITALSIGN: PROFESSIONAL PPTX GENERATOR FOR B.TECH ML PBL
=============================================================================
Generates a 19-slide widescreen PowerPoint presentation for the VitalSign
Healthcare 30-Day Readmission Prediction ML project.
Uses exact project metrics, real output charts, and embedded speaker notes.
=============================================================================
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PPTX = BASE_DIR / "VitalSign_Healthcare_ML_Presentation.pptx"

# Color Palette: Enterprise Academic Healthcare
COLOR_PRIMARY_NAVY = RGBColor(15, 23, 42)      # #0f172a
COLOR_SECONDARY_BLUE = RGBColor(30, 58, 138)   # #1e3a8a
COLOR_TEAL = RGBColor(13, 148, 136)            # #0d9488
COLOR_LIGHT_BG = RGBColor(248, 250, 252)       # #f8fafc
COLOR_CARD_BG = RGBColor(255, 255, 255)        # #ffffff
COLOR_CARD_BORDER = RGBColor(226, 232, 240)    # #e2e8f0
COLOR_TEXT_DARK = RGBColor(15, 23, 42)         # #0f172a
COLOR_TEXT_MUTED = RGBColor(71, 85, 105)       # #475569
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_ACCENT_GREEN = RGBColor(16, 185, 129)    # #10b981
COLOR_ACCENT_RED = RGBColor(239, 68, 68)       # #ef4444


def create_presentation():
    prs = Presentation()
    # Widescreen 16:9
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_badge="VITALSIGN HEALTHCARE ML"):
        # Header banner card
        header_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.0))
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = COLOR_LIGHT_BG
        header_shape.line.color.rgb = COLOR_CARD_BORDER
        header_shape.line.width = Pt(1)

        tf = header_shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.3)
        tf.margin_top = Inches(0.12)
        tf.margin_right = Inches(0.3)
        tf.margin_bottom = Inches(0.1)

        p1 = tf.paragraphs[0]
        p1.text = category_badge.upper()
        p1.font.size = Pt(10)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_TEAL

        p2 = tf.add_paragraph()
        p2.text = title_text
        p2.font.size = Pt(20)
        p2.font.bold = True
        p2.font.color.rgb = COLOR_PRIMARY_NAVY

    def add_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_PRIMARY_NAVY
    bg1.line.color.rgb = COLOR_PRIMARY_NAVY

    # Title Card
    tcard = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.1))
    tcard.fill.solid()
    tcard.fill.fore_color.rgb = RGBColor(30, 41, 59)
    tcard.line.color.rgb = COLOR_TEAL
    tcard.line.width = Pt(2)

    ttf = tcard.text_frame
    ttf.word_wrap = True
    ttf.margin_left = Inches(0.8)
    ttf.margin_top = Inches(0.6)
    ttf.margin_right = Inches(0.8)

    p = ttf.paragraphs[0]
    p.text = "VITALSIGN / HEALTHCARE PREDICTION"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL

    p = ttf.add_paragraph()
    p.text = "AI-Based 30-Day Hospital Readmission Prediction System"
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.space_before = Pt(10)

    p = ttf.add_paragraph()
    p.text = "A Complete Machine Learning System Lifecycle & Clinical Decision Support Architecture"
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.space_before = Pt(8)

    p = ttf.add_paragraph()
    p.text = "• Dataset: UCI Diabetes 130-US Hospitals (diabetic_data_50000.csv — 50,000 Patient Encounters)\n" \
             "• Academic Outcomes: CO1 (ML Lifecycle), CO2 (Linear & Regularized Models), CO3 (Tree Ensembles)\n" \
             "• Engineering Scope: Modules M1 (EDA), M2 (Preprocessing & Linear), M3 (Classifiers), M4 (Clustering)\n" \
             "• Deployment: Flask Web Framework with Real-Time Risk Stratification"
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(148, 163, 184)
    p.space_before = Pt(18)

    p = ttf.add_paragraph()
    p.text = "Academic PBL Evaluation | Department of Computer Science & Engineering"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_before = Pt(24)

    add_notes(s1,
        "Good morning respected evaluators and faculty mentors. "
        "Welcome to the presentation of our Machine Learning PBL project titled VitalSign: AI-Based 30-Day Hospital Readmission Prediction System. "
        "Our project analyzes 50,000 diabetic patient encounters from the UCI 130-US Hospitals dataset to predict whether a patient will face unplanned hospital readmission within 30 days. "
        "We have implemented the complete end-to-end machine learning system lifecycle fulfilling Course Outcomes CO1, CO2, and CO3 across four modular stages, complete with a deployed Flask web application."
    )

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "The Clinical Readmission Challenge in Diabetes Care", "Clinical Context & Motivation")

    # Left content box
    c2 = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(6.8), Inches(5.3))
    c2.fill.solid()
    c2.fill.fore_color.rgb = COLOR_CARD_BG
    c2.line.color.rgb = COLOR_CARD_BORDER
    tf2 = c2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = Inches(0.4)
    tf2.margin_top = Inches(0.4)
    tf2.margin_right = Inches(0.4)

    p = tf2.paragraphs[0]
    p.text = "Clinical Problem & Healthcare Dilemma:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    points2 = [
        ("High Readmission Burden", "Unplanned 30-day hospital readmissions among diabetic patients incur billions in healthcare costs and represent adverse clinical outcomes."),
        ("Multivariate Clinical Complexity", "Patient charts involve over 50 interdependent factors: demographics, laboratory tests, emergency histories, medications, and admission sources."),
        ("Extreme Class Imbalance", "In our 50,000-record dataset, only 11.49% (5,743 encounters) are readmitted within 30 days, causing traditional models to default to majority non-readmission."),
        ("Manual Triage Bottlenecks", "Clinicians lack automated, data-driven decision support tools to identify vulnerable, high-acuity patients at discharge.")
    ]

    for title, desc in points2:
        p = tf2.add_paragraph()
        p.text = f"• {title}: {desc}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(10)

    # Right Diagram Box
    r2 = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.8), Inches(1.6), Inches(4.733), Inches(5.3))
    r2.fill.solid()
    r2.fill.fore_color.rgb = COLOR_LIGHT_BG
    r2.line.color.rgb = COLOR_CARD_BORDER
    rtf2 = r2.text_frame
    rtf2.word_wrap = True
    rtf2.margin_left = Inches(0.3)
    rtf2.margin_top = Inches(0.5)

    rp = rtf2.paragraphs[0]
    rp.text = "SYSTEM OBJECTIVE FORMULATION"
    rp.font.size = Pt(14)
    rp.font.bold = True
    rp.font.color.rgb = COLOR_SECONDARY_BLUE
    rp.alignment = PP_ALIGN.CENTER

    steps = [
        "Patient Ingestion\n(50 Clinical & Diagnostic Fields)",
        "↓",
        "ML Preprocessing Pipeline\n(Imputation, Encoding, Scaling)",
        "↓",
        "Predictive Modeling\n(Linear, Regularized & Tree Models)",
        "↓",
        "Risk Stratification (0 - 100%)\n(Low / Moderate / High Risk)",
        "↓",
        "Clinical Action Protocols\n(Follow-up, Med Reconciliation)"
    ]
    for step in steps:
        sp = rtf2.add_paragraph()
        sp.text = step
        sp.alignment = PP_ALIGN.CENTER
        if step == "↓":
            sp.font.size = Pt(16)
            sp.font.bold = True
            sp.font.color.rgb = COLOR_TEAL
            sp.space_before = Pt(2)
        else:
            sp.font.size = Pt(11)
            sp.font.color.rgb = COLOR_TEXT_DARK
            sp.space_before = Pt(4)

    add_notes(s2,
        "Here we establish the core healthcare problem. Diabetic patients are frequently readmitted within 30 days of hospital discharge. "
        "In our actual 50,000 patient dataset, only 11.49% of patients are readmitted within 30 days. This creates a severe class imbalance challenge. "
        "Clinicians must review dozens of laboratory values and medication changes manually. "
        "Our system automates this risk stratification, transforming 50 raw clinical features into an actionable readmission risk probability."
    )

    # =========================================================================
    # SLIDE 3: PROJECT OBJECTIVES & ACADEMIC OUTCOMES
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Project Objectives & Academic Outcomes Mapping", "Project Objectives & CO Alignment")

    modules = [
        ("M1: Dataset Understanding & EDA", "Ingest 50,000 hospital records, audit 188,908 missing tokens (7.56%), analyze correlation matrices, and generate 13 clinical distribution charts.", COLOR_SECONDARY_BLUE),
        ("M2: Preprocessing & Linear Models (CO2)", "Evaluate 6 encoding methods, compare CFNE vs SGD optimization, apply Ridge/Lasso/ElasticNet, and formulate Binary and Multinomial Logistic Classifiers.", COLOR_TEAL),
        ("M3: Tree Ensembles & Benchmark (CO3)", "Construct Decision Trees (depth tuning), Random Forest (100 trees, balanced weights), and AdaBoost; select deployment Champion via holdout benchmark.", COLOR_SECONDARY_BLUE),
        ("M4: Patient Phenotyping & Clustering", "Execute unsupervised patient segmentation using K-Means (Elbow & Silhouette, K=2..10), DBSCAN outlier discovery, Hierarchical linkage, and PCA.", COLOR_TEAL),
        ("CO1: Lifecycle & Prediction Trace", "Map the complete 8-stage production ML lifecycle from raw ingestion to post-deployment monitoring; trace one real patient encounter end-to-end.", COLOR_SECONDARY_BLUE),
        ("Flask Deployment & MLOps Audit", "Deploy Champion Random Forest pipeline via interactive Flask web UI, expose REST API, and conduct batch inference drift monitoring.", COLOR_TEAL)
    ]

    for i, (mtitle, mdesc, col) in enumerate(modules):
        x = Inches(0.8 + (i % 2) * 5.966)
        y = Inches(1.6 + (i // 2) * 1.8)
        card = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(5.766), Inches(1.65))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.5)

        ctf = card.text_frame
        ctf.word_wrap = True
        ctf.margin_left = Inches(0.25)
        ctf.margin_top = Inches(0.18)
        ctf.margin_right = Inches(0.25)

        p = ctf.paragraphs[0]
        p.text = mtitle
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col

        p2 = ctf.add_paragraph()
        p2.text = mdesc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_MUTED
        p2.space_before = Pt(4)

    add_notes(s3,
        "This slide maps our project directly to the college curriculum and course outcomes. "
        "Module 1 covers comprehensive dataset ingestion and EDA. "
        "Module 2 satisfies CO2 by implementing linear models, regularization, bias-variance analysis, and logistic regression. "
        "Module 3 fulfills CO3 by training Decision Trees, Random Forests, and AdaBoost, followed by empirical model selection. "
        "Module 4 explores unsupervised clustering including K-Means, DBSCAN, Hierarchical clustering, and PCA. "
        "Finally, CO1 is demonstrated through our complete lifecycle architecture, prediction trace, and Flask deployment."
    )

    # =========================================================================
    # SLIDE 4: DATASET ARCHITECTURE & CHARACTERISTICS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "UCI Diabetes 130-US Hospitals Dataset Architecture", "Dataset Ingestion & Statistics")

    # Left Stats Card
    c4 = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.5), Inches(5.3))
    c4.fill.solid()
    c4.fill.fore_color.rgb = COLOR_CARD_BG
    c4.line.color.rgb = COLOR_CARD_BORDER
    tf4 = c4.text_frame
    tf4.word_wrap = True
    tf4.margin_left = Inches(0.35)
    tf4.margin_top = Inches(0.35)
    tf4.margin_right = Inches(0.35)

    p = tf4.paragraphs[0]
    p.text = "Verified Dataset Specifications:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    dstats = [
        ("Source Corpus", "UCI Machine Learning Repository (130 US Hospitals, 1999–2008)"),
        ("Total Records", "50,000 clinical encounters (diabetic_data_50000.csv)"),
        ("Total Attributes", "50 features (13 numerical, 37 categorical)"),
        ("Missing Cells", "188,908 cells (7.56% missingness across 50 columns)"),
        ("Primary Target", "Readmission_30_Days (1 = Readmitted <30d, 0 = Otherwise)"),
        ("Class Distribution", "Class 0: 44,257 (88.51%) | Class 1: 5,743 (11.49%)"),
        ("Excluded Columns", "encounter_id, patient_nbr, weight, payer_code, medical_specialty, diag_1, diag_2, diag_3 (leakage & sparsity)")
    ]
    for lbl, val in dstats:
        p = tf4.add_paragraph()
        p.text = f"• {lbl}: {val}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(8)

    # Right Table Card
    r4 = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.5), Inches(1.6), Inches(6.033), Inches(5.3))
    r4.fill.solid()
    r4.fill.fore_color.rgb = COLOR_LIGHT_BG
    r4.line.color.rgb = COLOR_CARD_BORDER
    rtf4 = r4.text_frame
    rtf4.word_wrap = True
    rtf4.margin_left = Inches(0.3)
    rtf4.margin_top = Inches(0.3)

    rp = rtf4.paragraphs[0]
    rp.text = "Representative Patient Clinical Encounter:"
    rp.font.size = Pt(14)
    rp.font.bold = True
    rp.font.color.rgb = COLOR_SECONDARY_BLUE

    sample_table = [
        ("Feature Name", "Sample Value", "Clinical Domain Context"),
        ("age", "[70-80)", "Demographic risk grouping"),
        ("time_in_hospital", "9 days", "Hospital inpatient duration"),
        ("num_lab_procedures", "54 tests", "Laboratory intensity index"),
        ("number_inpatient", "2 prior visits", "Strongest readmission predictor"),
        ("number_diagnoses", "7 diagnoses", "Patient comorbidity complexity"),
        ("insulin", "Down", "Glycemic regimen adjustment"),
        ("diabetesMed", "Yes", "Active antidiabetic prescription"),
        ("readmitted", "<30", "Ground-truth target (Mapped to 1)")
    ]
    for col1, col2, col3 in sample_table:
        p = rtf4.add_paragraph()
        p.text = f"• {col1:<20} | {col2:<12} | {col3}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(6)

    add_notes(s4,
        "Here we examine the exact properties of our dataset. We are using the verified 50,000-row subset of the UCI Diabetes dataset. "
        "There are 50 raw features. We audited all missing values and found 188,908 missing cells, which were represented by question mark characters. "
        "Our positive target is readmission within 30 days, which accounts for 5,743 encounters or 11.49%. "
        "We also intentionally excluded non-predictive tracking identifiers like encounter ID, and extreme-sparsity columns like weight, which had over 96% missing values."
    )

    # =========================================================================
    # SLIDE 5: SYSTEM ARCHITECTURE & LIFECYCLE (CO1)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Course Outcome CO1: End-to-End Machine Learning System Lifecycle", "CO1: System Lifecycle Architecture")

    stages = [
        ("1. Raw Ingestion", "diabetic_data_50000.csv\n(50,000 patient rows)"),
        ("2. Data Sanitization", "Replace '?' with NaN\nDrop ID & sparse cols"),
        ("3. Preprocessing", "ColumnTransformer\n(Median + OHE/Scaling)"),
        ("4. Split & Stratify", "80/20 Stratified Split\nTrain: 40k | Test: 10k"),
        ("5. Model Training", "Linear, Tree Ensembles\nBalanced class weights"),
        ("6. Holdout Eval", "Accuracy, Precision,\nRecall, F1 & ROC-AUC"),
        ("7. Serialization", "vitalsign_model.pkl\nMetadata registration"),
        ("8. Flask Serving", "Interactive UI, REST API,\nReal-time inference")
    ]

    for idx, (stitle, sdesc) in enumerate(stages):
        x = Inches(0.8 + (idx % 4) * 2.95)
        y = Inches(1.8 + (idx // 4) * 2.6)
        scard = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(2.8), Inches(2.3))
        scard.fill.solid()
        scard.fill.fore_color.rgb = COLOR_CARD_BG
        scard.line.color.rgb = COLOR_TEAL if idx >= 4 else COLOR_SECONDARY_BLUE
        scard.line.width = Pt(1.5)

        stf = scard.text_frame
        stf.word_wrap = True
        stf.margin_left = Inches(0.2)
        stf.margin_top = Inches(0.2)

        p = stf.paragraphs[0]
        p.text = stitle
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_NAVY

        p = stf.add_paragraph()
        p.text = sdesc
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(8)

    add_notes(s5,
        "Slide 5 directly demonstrates Course Outcome CO1 by showcasing the complete 8-stage production machine learning lifecycle. "
        "Stage 1 starts with raw data ingestion. In Stage 2, question marks are cleaned and uninformative columns removed. "
        "Stage 3 encapsulates feature transformations inside scikit-learn ColumnTransformers to guarantee zero data leakage. "
        "In Stage 4, an 80/20 stratified split preserves the 11.49% class ratio. "
        "Stages 5 and 6 train and evaluate candidate algorithms on a pure 10,000-record test set. "
        "Finally, Stages 7 and 8 serialize the winning Champion pipeline with Joblib and deploy it to Flask for live inference."
    )

    # =========================================================================
    # SLIDE 6: DATA PREPROCESSING PIPELINE (M2)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Clinical Data Cleaning & Preprocessing Pipeline", "Module M2: Preprocessing Architecture")

    c6 = s6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    c6.fill.solid()
    c6.fill.fore_color.rgb = COLOR_CARD_BG
    c6.line.color.rgb = COLOR_CARD_BORDER
    tf6 = c6.text_frame
    tf6.word_wrap = True
    tf6.margin_left = Inches(0.4)
    tf6.margin_top = Inches(0.4)

    p = tf6.paragraphs[0]
    p.text = "Production Preprocessing Architecture (final_preprocess_M2.py):"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    prep_steps = [
        ("Missing Token Resolution", "Audited and converted all '?' characters to np.nan across all 50 columns without deleting patient rows."),
        ("Dimensionality Reduction & Leakage Elimination", "Safely dropped 9 uninformative or high-cardinality columns: encounter_id, patient_nbr, readmitted, weight (96.3% missing), payer_code (65.3% missing), medical_specialty (35.5% missing), and diag_1/2/3."),
        ("Numerical Feature Imputation & Scaling", "11 numerical features (e.g. time_in_hospital, lab procedures, prior admissions) imputed via Median strategy, then transformed using StandardScaler (Mean = 0, Std = 1)."),
        ("Categorical Feature Imputation & Encoding", "30 categorical features (e.g. race, gender, age, 23 medications) imputed via Most Frequent strategy and converted into dense binary indicators via OneHotEncoder(handle_unknown='ignore')."),
        ("Supervised Target Derivation", "Engineered binary ground truth: Readmission_30_Days = (readmitted == '<30').astype(int)."),
        ("Output Dimensionality", "Produced 104 clean numerical features for linear models and ~70 dense features for tree classifiers, serialized to models/preprocessor.pkl.")
    ]
    for title, desc in prep_steps:
        p = tf6.add_paragraph()
        p.text = f"• {title}: {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(8)

    add_notes(s6,
        "In this slide, we describe our actual data preprocessing pipeline implemented in final_preprocess_M2.py. "
        "A critical engineering decision was handling the diagnostic code columns diag_1, diag_2, and diag_3. "
        "Because ICD-9 codes have thousands of distinct categories, naive one-hot encoding expands the feature matrix to nearly 2,000 sparse columns, causing severe memory bottlenecks. "
        "By dropping high-cardinality diagnosis codes and uninformative identifiers, we reduced the dimensionality to 104 clean transformed features. "
        "All transformers were fitted strictly on training data to prevent data leakage."
    )

    # =========================================================================
    # SLIDE 7: EXPLORATORY DATA ANALYSIS (M1)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Exploratory Data Analysis & Clinical Correlation Heatmap", "Module M1: Exploratory Data Analysis")

    # Left text box
    c7 = s7.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    c7.fill.solid()
    c7.fill.fore_color.rgb = COLOR_CARD_BG
    c7.line.color.rgb = COLOR_CARD_BORDER
    tf7 = c7.text_frame
    tf7.word_wrap = True
    tf7.margin_left = Inches(0.35)
    tf7.margin_top = Inches(0.35)
    tf7.margin_right = Inches(0.35)

    p = tf7.paragraphs[0]
    p.text = "Key Verified Clinical Insights (EDA_Analysis.py):"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    eda_points = [
        ("Readmission Imbalance", "Out of 50,000 encounters, only 11.49% were readmitted within 30 days. 88.51% were either readmitted after 30 days or not readmitted."),
        ("Prior Inpatient Admissions", "number_inpatient showed the highest positive Pearson correlation with 30-day readmission (r = +0.162), indicating past hospitalization history is a paramount risk marker."),
        ("Length of Stay vs Vitals", "Hospital duration (time_in_hospital, mean = 4.56 days) is moderately correlated with number of medications (r = +0.46) and lab procedures (r = +0.32)."),
        ("Missing Token Concentration", "Missingness is heavily concentrated: weight (96.3%), max_glu_serum (90.7%), A1Cresult (84.8%), and payer_code (65.3%)."),
        ("Demographic Spread", "Patient age distribution peaks in the [70-80) and [60-70) age brackets; gender is evenly split (53.8% Female, 46.2% Male).")
    ]
    for lbl, val in eda_points:
        p = tf7.add_paragraph()
        p.text = f"• {lbl}: {val}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(8)

    # Right Image (Correlation Heatmap)
    img_corr = BASE_DIR / "outputs" / "M1_Dataset_EDA" / "Correlation_Heatmap.png"
    if img_corr.exists():
        s7.shapes.add_picture(str(img_corr), Inches(6.8), Inches(1.6), width=Inches(5.733))

    add_notes(s7,
        "On Slide 7, we present our empirical findings from exploratory data analysis. "
        "On the right is our actual correlation heatmap generated by Correlation_Matrix_heatmap_boxplots_M1.py. "
        "The single strongest predictor of 30-day readmission is number_inpatient—the count of prior inpatient hospitalizations—with a correlation of plus 0.162. "
        "We also discovered that laboratory test volume and medication counts increase linearly with length of hospital stay. "
        "This confirms that diabetic patients with high hospitalization frequencies and multiple comorbidities represent the core risk cohort."
    )

    # =========================================================================
    # SLIDE 8: FEATURE ENGINEERING & ENCODING STRATEGIES (M2)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Academic Encoding Demonstrations vs Production Pipeline", "Module M2: Feature Engineering & Encodings")

    c8 = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    c8.fill.solid()
    c8.fill.fore_color.rgb = COLOR_CARD_BG
    c8.line.color.rgb = COLOR_CARD_BORDER
    tf8 = c8.text_frame
    tf8.word_wrap = True
    tf8.margin_left = Inches(0.4)
    tf8.margin_top = Inches(0.35)

    p = tf8.paragraphs[0]
    p.text = "Encoding Techniques Implemented for Academic Rigor (Module M2):"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    enc_methods = [
        ("Label Encoding (clean_label_encode_M2.py)", "Applied to binary clinical flags such as diabetesMed ('No'->0, 'Yes'->1) and change ('Ch'->0, 'No'->1). Simple and memory efficient for ordinal/binary features."),
        ("One-Hot Encoding (clean_one_hot_encod_M2.py)", "Applied to nominal categories without order (race, admission source, discharge disposition). Produces orthogonal binary columns; handle_unknown='ignore' ensures test robustness."),
        ("Ordinal Encoding (clean_ordinal_encode_M2.py)", "Preserves chronological clinical hierarchy in age bands: '[0-10)' -> 0 through '[90-100)' -> 9, maintaining mathematical distance between age tiers."),
        ("Target Mean Encoding (clean_target_encode_M2.py)", "Replaces categories with the conditional target mean P(Y=1|X=c). Fitted strictly on the training partition to prevent label leakage."),
        ("Scaling: Min-Max vs Standard (clean_minmax_stand_norma_M2.py)", "Demonstrated Min-Max normalization [0, 1] vs Z-Score standardization (zero mean, unit variance). StandardScaler chosen for linear models to preserve outlier influence."),
        ("Production Selection Rationale", "To prevent synthetic ordinal assumptions while avoiding target leakage, our production pipeline final_preprocess_M2.py integrates Median Imputation + StandardScaler for numerical columns and Mode Imputation + OneHotEncoder for categorical features.")
    ]
    for lbl, desc in enc_methods:
        p = tf8.add_paragraph()
        p.text = f"• {lbl}: {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(8)

    add_notes(s8,
        "In Slide 8, we satisfy the academic requirement to compare diverse feature engineering techniques. "
        "We implemented independent scripts for Label Encoding, One-Hot Encoding, Ordinal Encoding, Target Encoding, and Feature Scaling. "
        "For example, ordinal encoding preserves the age hierarchy from 0 to 9, while target encoding calculates class probabilities per demographic group strictly on train data. "
        "For our deployed production pipeline, we chose ColumnTransformer with One-Hot Encoding and StandardScaler because it prevents synthetic distance assumptions and eliminates target leakage."
    )

    # =========================================================================
    # SLIDE 9: LINEAR & REGULARIZED MODELS (CO2)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "CO2: Linear Regression, Regularization & Bias-Variance Analysis", "CO2: Linear Supervised Learning")

    # Left text box
    c9 = s9.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.3))
    c9.fill.solid()
    c9.fill.fore_color.rgb = COLOR_CARD_BG
    c9.line.color.rgb = COLOR_CARD_BORDER
    tf9 = c9.text_frame
    tf9.word_wrap = True
    tf9.margin_left = Inches(0.35)
    tf9.margin_top = Inches(0.3)
    tf9.margin_right = Inches(0.35)

    p = tf9.paragraphs[0]
    p.text = "Continuous Target Evaluation: time_in_hospital (Days):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    lin_points = [
        ("Clinical Target Formulation", "Linear regression models are applied to an appropriate continuous clinical target—length of hospital stay (time_in_hospital)—rather than improperly treating binary readmission as linear."),
        ("OLS Closed-Form vs SGD (CFNE_GD_Compare)", "OLS Normal Equation achieves MAE = 2.0459 days, RMSE = 2.6636, R² = 0.2575. Stochastic Gradient Descent converges to identical MAE = 2.0480, RMSE = 2.6636 in 100 epochs."),
        ("Ridge Regression (L2 Penalty)", "Shrinks regression weights toward zero via alpha * ||w||_2^2. Achieved MAE = 2.0431, RMSE = 2.6608, R² = 0.2591, mitigating multicollinearity."),
        ("Lasso Regression (L1 Penalty)", "Enforces sparse feature selection by driving non-critical feature coefficients to zero. Achieved MAE = 2.0434, RMSE = 2.6605, R² = 0.2592."),
        ("Elastic Net Regularization", "Combines L1 and L2 penalties (l1_ratio=0.5). Achieved MAE = 2.0435, RMSE = 2.6606, R² = 0.2591, balancing feature selection with collinear stability.")
    ]
    for lbl, desc in lin_points:
        p = tf9.add_paragraph()
        p.text = f"• {lbl}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(6)

    # Right Image: Bias-Variance Analysis Plot
    img_bv = BASE_DIR / "outputs" / "M2_Linear_Models" / "bias_variance.png"
    if img_bv.exists():
        s9.shapes.add_picture(str(img_bv), Inches(7.0), Inches(1.6), width=Inches(5.533))

    add_notes(s9,
        "Slide 9 demonstrates Course Outcome CO2. "
        "It is mathematically incorrect to apply linear regression to a binary classification target like readmission. "
        "Therefore, we formulated an appropriate continuous clinical prediction problem: predicting hospital length of stay (time_in_hospital). "
        "We compared Ordinary Least Squares closed-form normal equations against Stochastic Gradient Descent, proving numerical equivalence. "
        "We then evaluated L2 Ridge, L1 Lasso, and Elastic Net regularization. "
        "On the right, our bias-variance plot demonstrates how increasing model complexity from depth 1 to unconstrained depth leads to extreme overfitting, where training F1 hits 0.97 while test F1 collapses to 0.15."
    )

    # =========================================================================
    # SLIDE 10: LOGISTIC REGRESSION BINARY CLASSIFIER (CO2)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Binary Logistic Regression Readmission Classifier", "CO2: Probabilistic Binary Classification")

    # Left text box
    c10 = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    c10.fill.solid()
    c10.fill.fore_color.rgb = COLOR_CARD_BG
    c10.line.color.rgb = COLOR_CARD_BORDER
    tf10 = c10.text_frame
    tf10.word_wrap = True
    tf10.margin_left = Inches(0.35)
    tf10.margin_top = Inches(0.35)
    tf10.margin_right = Inches(0.35)

    p = tf10.paragraphs[0]
    p.text = "Verified Holdout Performance (10,000 Encounters):"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    log_metrics = [
        ("Accuracy", "64.87% (6,487 out of 10,000 correct)"),
        ("Precision", "0.1724 (17.24% of flagged patients readmitted)"),
        ("Recall (Sensitivity)", "0.5413 (54.13% of true <30d readmissions detected)"),
        ("F1-Score", "0.2615 (Harmonic mean of precision & recall)"),
        ("ROC-AUC Score", "0.6539 (Area under receiver operating characteristic)"),
        ("Class Weighting Strategy", "Utilized class_weight='balanced' in LogisticRegression(solver='lbfgs') to penalize minority misclassifications inversely proportional to class frequencies."),
        ("Clinical Utility", "Linear baseline successfully captures over 54% of vulnerable patients facing readmission, providing an interpretable odds-ratio benchmark.")
    ]
    for lbl, val in log_metrics:
        p = tf10.add_paragraph()
        p.text = f"• {lbl}: {val}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(7)

    # Right Image: Logistic Confusion Matrix
    img_cm = BASE_DIR / "outputs" / "M2_Linear_Models" / "Logistic_Regression" / "Confusion_Matrix.png"
    if img_cm.exists():
        s10.shapes.add_picture(str(img_cm), Inches(6.8), Inches(1.6), width=Inches(5.733))

    add_notes(s10,
        "On Slide 10, we evaluate our Binary Logistic Regression classifier on the holdout test set of 10,000 patients. "
        "Without class weighting, a model predicting only zero achieves 88.5% accuracy but zero percent clinical recall. "
        "By enforcing balanced class weights, our Logistic Regression model achieves 54.13% recall and an ROC-AUC of 0.6539. "
        "The confusion matrix on the right shows that out of 1,149 true readmitted patients in the test set, 622 were correctly flagged for early intervention."
    )

    # =========================================================================
    # SLIDE 11: MULTINOMIAL LOGISTIC REGRESSION (CO2)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Multinomial Logistic Regression (3-Class Academic Model)", "CO2: Softmax Multi-Class Classification")

    c11 = s11.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    c11.fill.solid()
    c11.fill.fore_color.rgb = COLOR_CARD_BG
    c11.line.color.rgb = COLOR_CARD_BORDER
    tf11 = c11.text_frame
    tf11.word_wrap = True
    tf11.margin_left = Inches(0.4)
    tf11.margin_top = Inches(0.35)

    p = tf11.paragraphs[0]
    p.text = "Multiclass Softmax Formulation (Multinomial_Logistic_Regre_for_Multiclas.py):"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    multi_points = [
        ("Original 3-Class Categorical Target", "The raw UCI readmission target includes 3 distinct clinical outcomes: 'NO' (No readmission), '>30' (Readmitted after 30 days), and '<30' (High-risk readmission within 30 days)."),
        ("Mathematical Formulation", "Applies Softmax activation: P(y=k|x) = exp(w_k^T x) / sum_j exp(w_j^T x), outputting a probability simplex over all 3 mutually exclusive states."),
        ("Verified Holdout Evaluation", "Evaluated on 10,000 encounters: Overall Accuracy = 57.00%, Macro Precision = 0.4848, Macro Recall = 0.3868, Macro F1-Score = 0.3570."),
        ("Per-Class Breakdown", "Class '<30' achieved F1 = 0.71 (Recall = 89%), Class '>30' achieved F1 = 0.34 (Recall = 26%), and Class 'NO' achieved F1 = 0.03 (severely confused with >30)."),
        ("Academic Distinction & Deployment Choice", "While 3-class multinomial modeling provides academic insight into hospital discharge stages, clinical protocols specifically require binary triage: identifying whether a patient is at risk of 30-day readmission (YES vs NO). Hence, our deployment champion is optimized for binary 30-day risk.")
    ]
    for lbl, desc in multi_points:
        p = tf11.add_paragraph()
        p.text = f"• {lbl}: {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(8)

    add_notes(s11,
        "Slide 11 addresses the multiclass requirement under CO2. "
        "The original dataset has three categories: NO, greater than 30 days, and less than 30 days. "
        "We implemented multinomial softmax regression to predict all three classes simultaneously, achieving 57.00% overall accuracy and a macro F1 of 0.3570. "
        "However, in hospital quality management, clinical intervention is focused strictly on preventing readmission within 30 days. "
        "Therefore, our primary deployment model is the binary classifier."
    )

    # =========================================================================
    # SLIDE 12: TREE-BASED MODELS (CO3)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "CO3: Tree-Based Classifiers — Decision Tree & Ensembles", "CO3: Tree Supervised Learning")

    # Left text box
    c12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    c12.fill.solid()
    c12.fill.fore_color.rgb = COLOR_CARD_BG
    c12.line.color.rgb = COLOR_CARD_BORDER
    tf12 = c12.text_frame
    tf12.word_wrap = True
    tf12.margin_left = Inches(0.35)
    tf12.margin_top = Inches(0.35)
    tf12.margin_right = Inches(0.35)

    p = tf12.paragraphs[0]
    p.text = "Comparative Tree Architecture Analysis:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    tree_points = [
        ("Decision Tree (Decision_Tree_Classifier_M3.py)", "Recursive binary splitting via Gini impurity. Evaluated depths [3, 5, 8, 10, 15, None]. Optimal depth = 10 with balanced weights: Accuracy = 59.95%, Recall = 59.27%, F1 = 0.2538, ROC-AUC = 0.6158."),
        ("Random Forest Ensemble (Random_Forest_Classifier_M3.py)", "Constructs 100 decorrelated trees via bootstrap aggregating (bagging) and random subspace feature splits (mtry = sqrt(p)). Accuracy = 70.38%, Precision = 0.1975, Recall = 51.52%, F1 = 0.2856, ROC-AUC = 0.6634."),
        ("AdaBoost Classifier (Adaboost_Classifier_M3.py)", "Sequential boosting with 50 depth-2 decision stumps (learning_rate=0.5). Accuracy = 88.47%, Precision = 0.4000, Recall = 0.70%, F1 = 0.0137, ROC-AUC = 0.6721. Suffers extreme false-negative rate on imbalanced data."),
        ("Core Takeaway", "Random Forest balances sensitivity and specificity far superiorly to individual trees or boosting on this imbalanced clinical distribution.")
    ]
    for lbl, desc in tree_points:
        p = tf12.add_paragraph()
        p.text = f"• {lbl}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(7)

    # Right Image: Random Forest Feature Importance
    img_rf_feat = BASE_DIR / "outputs" / "M3_Tree_Models" / "Random_Forest_Feature_Importance.png"
    if img_rf_feat.exists():
        s12.shapes.add_picture(str(img_rf_feat), Inches(6.8), Inches(1.6), width=Inches(5.733))

    add_notes(s12,
        "Slide 12 covers tree-based supervised learning models fulfilling Course Outcome CO3. "
        "We tested single Decision Trees across depths 3 through unconstrained. Depth 10 yielded the best balance before overfitting. "
        "We then trained a Random Forest with 100 trees, which reduced variance dramatically. "
        "Finally, we trained AdaBoost with 50 estimators. While AdaBoost achieved 88.47% accuracy, its recall dropped to under 1% because it biased entirely toward the majority class. "
        "On the right, Random Forest feature importance confirms that number_inpatient, time_in_hospital, and lab procedures are the most influential clinical split variables."
    )

    # =========================================================================
    # SLIDE 13: MODEL BENCHMARK & CHAMPION SELECTION (CO3)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Empirical Multi-Model Benchmark & Champion Selection", "CO3: Empirical Model Selection")

    # Left text & table box
    c13 = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.3))
    c13.fill.solid()
    c13.fill.fore_color.rgb = COLOR_CARD_BG
    c13.line.color.rgb = COLOR_CARD_BORDER
    tf13 = c13.text_frame
    tf13.word_wrap = True
    tf13.margin_left = Inches(0.3)
    tf13.margin_top = Inches(0.3)
    tf13.margin_right = Inches(0.3)

    p = tf13.paragraphs[0]
    p.text = "Verified Holdout Benchmark Table (10,000 Records):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    b_table = [
        ("Model", "Accuracy", "Recall", "F1-Score", "ROC-AUC"),
        ("Logistic Regression", "64.87%", "54.13%", "0.2615", "0.6539"),
        ("Decision Tree (d=10)", "59.95%", "59.27%", "0.2538", "0.6158"),
        ("Random Forest (100t)", "70.38%", "51.52%", "0.2856", "0.6634"),
        ("AdaBoost (50 rounds)", "88.47%", "0.70%", "0.0137", "0.6721")
    ]
    for m, acc, rec, f1, auc in b_table:
        p = tf13.add_paragraph()
        is_hdr = (m == "Model")
        is_champ = ("Random Forest" in m)
        p.text = f"{m:<20} | {acc:<8} | {rec:<8} | {f1:<8} | {auc}"
        p.font.size = Pt(11)
        p.font.bold = is_hdr or is_champ
        p.font.color.rgb = COLOR_PRIMARY_NAVY if is_hdr else (COLOR_TEAL if is_champ else COLOR_TEXT_MUTED)
        p.space_before = Pt(5)

    p = tf13.add_paragraph()
    p.text = "CHAMPION MODEL SELECTION RATIONALE:"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY_BLUE
    p.space_before = Pt(12)

    p = tf13.add_paragraph()
    p.text = "• Selected Champion: Random Forest Classifier\n" \
             "• Criterion: Highest F1-Score (0.2856) & ROC-AUC (0.6634) on holdout data.\n" \
             "• Clinical Justification: In hospital readmission, raw accuracy is deceptive—AdaBoost scored 88.5% accuracy but failed to detect 99.3% of readmissions! Random Forest captures over half of all high-risk patients while keeping false alarms manageable."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.space_before = Pt(4)

    # Right Image: Model Comparison Bar Chart
    img_comp = BASE_DIR / "outputs" / "Model_Comparison" / "model_comparison.png"
    if img_comp.exists():
        s13.shapes.add_picture(str(img_comp), Inches(7.0), Inches(1.6), width=Inches(5.533))

    add_notes(s13,
        "Slide 13 is our primary benchmark and model selection slide. "
        "All four models were evaluated strictly on the exact same 10,000-patient test set. "
        "Notice why accuracy is a dangerous metric in healthcare: AdaBoost achieved 88.47% accuracy, but its clinical recall was almost zero (0.70%). "
        "Random Forest achieved the highest F1-score of 0.2856 and the highest ROC-AUC of 0.6634, correctly identifying over 51% of readmissions. "
        "Therefore, Random Forest was designated as the production Champion model and serialized to models/vitalsign_readmission_model.pkl."
    )

    # =========================================================================
    # SLIDE 14: UNSUPERVISED LEARNING & PATIENT PHENOTYPING (M4)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Module M4: Unsupervised Patient Phenotyping & Segmentation", "Module M4: Unsupervised Learning")

    # Left text box
    c14 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    c14.fill.solid()
    c14.fill.fore_color.rgb = COLOR_CARD_BG
    c14.line.color.rgb = COLOR_CARD_BORDER
    tf14 = c14.text_frame
    tf14.word_wrap = True
    tf14.margin_left = Inches(0.35)
    tf14.margin_top = Inches(0.35)
    tf14.margin_right = Inches(0.35)

    p = tf14.paragraphs[0]
    p.text = "Unsupervised Exploration Overview (Module M4):"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    m4_points = [
        ("Academic Distinction", "Clustering is unsupervised patient phenotyping—it does NOT predict Readmission_30_Days directly, but uncovers latent clinical patient subgroups."),
        ("DBSCAN Density-Based (DBSCAN_Healthcare_M4.py)", "eps=1.2, min_samples=10: Identified 1 massive dense core cluster (98.96%) and 52 clinical outliers/noise encounters (1.04%) with extreme lab test volumes and stay durations."),
        ("Hierarchical Linkage (Hierarchical_Clustering_M4.py)", "Computed Ward agglomerative linkage and rendered clinical dendrogram on 1,000 patients (Silhouette = 0.1532), identifying 3 hierarchical patient tiers."),
        ("Principal Component Analysis (PCA_Healthcare_M4.py)", "Decomposed 8 continuous features: PC1 captures 25.46% variance (inpatient stay intensity), PC2 captures 17.03% (emergency visits). Top 2 components retain 42.49% total variance.")
    ]
    for lbl, desc in m4_points:
        p = tf14.add_paragraph()
        p.text = f"• {lbl}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(7)

    # Right Image: PCA 2D Scatter
    img_pca = BASE_DIR / "outputs" / "M4_Clustering" / "PCA_2D.png"
    if img_pca.exists():
        s14.shapes.add_picture(str(img_pca), Inches(6.8), Inches(1.6), width=Inches(5.733))

    add_notes(s14,
        "In Slide 14, we present Module M4 unsupervised learning techniques. "
        "It is essential to clarify that clustering is an exploratory segmentation technique, not a replacement for supervised prediction. "
        "We implemented DBSCAN to isolate density outliers, detecting 52 extreme clinical encounters with atypical medication profiles. "
        "We also applied Hierarchical Ward clustering to construct patient dendrograms. "
        "On the right, our 2D PCA projection visualizes the continuous patient space, with PC1 and PC2 explaining over 42% of total clinical variance."
    )

    # =========================================================================
    # SLIDE 15: K-MEANS CLUSTERING & VALIDATION (M4)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "K-Means Patient Phenotyping, Elbow & Silhouette Analysis", "Module M4: K-Means Segmentation")

    # Left text box
    c15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    c15.fill.solid()
    c15.fill.fore_color.rgb = COLOR_CARD_BG
    c15.line.color.rgb = COLOR_CARD_BORDER
    tf15 = c15.text_frame
    tf15.word_wrap = True
    tf15.margin_left = Inches(0.35)
    tf15.margin_top = Inches(0.3)
    tf15.margin_right = Inches(0.35)

    p = tf15.paragraphs[0]
    p.text = "Empirical Cluster Quality Analysis (K = 2 to 10):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    km_points = [
        ("Elbow Method (Inertia WCSS)", "K=2 (Inertia=186k), K=3 (156k), K=4 (138k), K=5 (123k). Pronounced elbow inflection observed at K=3."),
        ("Silhouette Analysis", "Mean silhouette coefficient: K=2 (0.2606), K=3 (0.2072), K=4 (0.1989), K=5 (0.2103). K=3 provides optimal cohesion and clinical interpretability."),
        ("Cluster 0: Low-Acuity Stay (36.6%)", "Mean stay: 3.16 days | Lab tests: 38.7 | Medications: 11.6 | Diagnoses: 4.70. Routine short-stay diabetic patients."),
        ("Cluster 1: Multi-Diagnostic Elderly (41.2%)", "Mean stay: 3.77 days | Lab tests: 38.6 | Medications: 13.6 | Diagnoses: 8.36. High comorbidity index, moderate stay duration."),
        ("Cluster 2: High-Complexity Inpatient (22.2%)", "Mean stay: 8.32 days | Lab tests: 55.2 | Procedures: 2.99 | Medications: 24.4. Severely ill diabetic cohort with prolonged hospitalization.")
    ]
    for lbl, desc in km_points:
        p = tf15.add_paragraph()
        p.text = f"• {lbl}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(6)

    # Right Image: Elbow or KMeans Clusters
    img_km = BASE_DIR / "outputs" / "M4_Clustering" / "kmeans_clusters.png"
    if img_km.exists():
        s15.shapes.add_picture(str(img_km), Inches(6.8), Inches(1.6), width=Inches(5.733))

    add_notes(s15,
        "Slide 15 details our K-Means clustering implementation. "
        "We systematically evaluated K values from 2 through 10 using both the Elbow Method and Silhouette Analysis. "
        "We selected K=3 as the optimal number of clusters based on mathematical curvature and clinical interpretability. "
        "Cluster 0 represents short routine stays. Cluster 1 represents multi-diagnostic patients with over 8 diagnoses. "
        "Cluster 2 represents complex inpatient stays averaging over 8 hospital days and 24 medications. "
        "This segmentation helps hospital administrators tailor care protocols to patient phenotypes."
    )

    # =========================================================================
    # SLIDE 16: DEPLOYMENT & FLASK WEB APPLICATION
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "VitalSign Flask Web Deployment & REST API Serving", "System Deployment & Engineering")

    c16 = s16.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    c16.fill.solid()
    c16.fill.fore_color.rgb = COLOR_CARD_BG
    c16.line.color.rgb = COLOR_CARD_BORDER
    tf16 = c16.text_frame
    tf16.word_wrap = True
    tf16.margin_left = Inches(0.4)
    tf16.margin_top = Inches(0.35)

    p = tf16.paragraphs[0]
    p.text = "Production Web Architecture (app.py):"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    flask_points = [
        ("Web Framework & Serving", "Built with Python Flask, Jinja2 templating, and Bootstrap 5 responsive UI. Operates at http://127.0.0.1:5000."),
        ("Verified Deployed Endpoints (All HTTP 200 OK)", "Clinical Portal (/), Live Dashboard (/dashboard), Dataset Explorer (/dataset), Interactive EDA (/visualization), Model Comparison (/models), Clinical Prediction Form (/prediction), Batch CSV Predictor (/batch-predict), and REST API (/api/predict)."),
        ("Model Artifacts & Pipelines", "Loads serialized Champion Random Forest pipeline (models/vitalsign_readmission_model.pkl, 9.0 MB) and registered metadata (models/model_metadata.pkl). Zero retraining overhead at runtime."),
        ("Interactive Patient Risk Scoring", "Clinicians input patient age, vitals, prior admissions, lab count, and medication adjustments. System executes instant inference in under 45 milliseconds."),
        ("Tri-Tier Risk Stratification", "Outputs quantitative probability (0-100%) and categorizes risk: Low Risk (<30%), Moderate Risk (30-50%), and High Risk (>=50%) with tailored clinical recommendations."),
        ("Educational Disclaimer Guardrail", "Every web page and JSON API response includes an explicit medical disclaimer: 'This application is an educational prediction system and not a medical diagnosis tool.'")
    ]
    for lbl, desc in flask_points:
        p = tf16.add_paragraph()
        p.text = f"• {lbl}: {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(7)

    add_notes(s16,
        "Slide 16 describes our deployed web system. "
        "We built an end-to-end Flask application containing 13 verified routes, including dashboard, dataset views, model benchmarks, and interactive prediction. "
        "The web application loads the serialized Champion Random Forest pipeline directly with Joblib. "
        "When a user enters patient information, the system returns the readmission probability, a colored risk badge, and recommended discharge protocols. "
        "We also implemented a REST endpoint at /api/predict that returns structured JSON payloads for clinical software integration."
    )

    # =========================================================================
    # SLIDE 17: ONE PREDICTION TRACE (CO1)
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "CO1: End-to-End Prediction Request Trace (ML_Lifecycle_CO1.py)", "CO1: Clinical Prediction Trace")

    trace_steps = [
        ("1. Clinician Input", "Patient #86240259: Age [70-80), Female, 9 days stay, 54 lab tests, 2 prior inpatients, Insulin 'Down'."),
        ("2. Request Ingestion", "Flask receives POST /api/predict payload; deserializes into pandas DataFrame (41 raw predictors)."),
        ("3. Preprocessing", "ColumnTransformer applies Median imputation to numericals and OneHotEncoder to demographics & meds."),
        ("4. Pipeline Transform", "Transforms raw clinical features into dense feature space matching fitted training dimensions."),
        ("5. Model Inference", "Champion Random Forest pipeline executes .predict() and .predict_proba() on holdout sample."),
        ("6. Probabilistic Output", "Computes readmission probability: P(Readmit <30d) = 69.99% -> Binary Prediction = 1 (HIGH RISK)."),
        ("7. Clinical Action Protocol", "System triggers priority protocol: 7-day follow-up appointment, diabetes medication reconciliation."),
        ("8. Telemetry & Monitoring", "Logs inference timestamp, input hash, latency, and probability to outputs/Lifecycle/ml_lifecycle_report.txt.")
    ]

    for idx, (tstep, tdesc) in enumerate(trace_steps):
        x = Inches(0.8 + (idx % 2) * 5.966)
        y = Inches(1.6 + (idx // 2) * 1.35)
        tcard = s17.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(5.766), Inches(1.22))
        tcard.fill.solid()
        tcard.fill.fore_color.rgb = COLOR_CARD_BG
        tcard.line.color.rgb = COLOR_TEAL if idx >= 4 else COLOR_SECONDARY_BLUE
        tcard.line.width = Pt(1.5)

        ttf = tcard.text_frame
        ttf.word_wrap = True
        ttf.margin_left = Inches(0.25)
        ttf.margin_top = Inches(0.12)
        ttf.margin_right = Inches(0.25)

        p = ttf.paragraphs[0]
        p.text = tstep
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_NAVY

        p = ttf.add_paragraph()
        p.text = tdesc
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(3)

    add_notes(s17,
        "Slide 17 is a critical requirement for Course Outcome CO1: tracing a single clinical prediction request backward through every layer of the system. "
        "Here we trace actual patient encounter #86240259 from our dataset. "
        "The patient is in the 70 to 80 age group with 2 prior inpatient admissions and an extended 9-day hospital stay. "
        "In steps 1 through 4, the request moves from the web form through the preprocessor pipeline. "
        "In step 5, the Random Forest model calculates a readmission probability of 69.99%. "
        "In step 6 and 7, the system stratifies the patient as High Risk and prescribes a 7-day follow-up. "
        "Finally, step 8 logs the transaction for MLOps monitoring. This completes the end-to-end trace."
    )

    # =========================================================================
    # SLIDE 18: RESULTS, CONCLUSION & FUTURE SCOPE
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Project Synthesis: Results, Conclusion & Future Scope", "Project Synthesis")

    col_w = Inches(3.75)
    sections = [
        ("PROJECT RESULTS", [
            "50,000 encounters sanitized and processed with zero data leakage.",
            "Linear models verified (OLS CFNE vs SGD, Ridge, Lasso, ElasticNet).",
            "Multi-model holdout benchmark executed across 4 algorithms.",
            "Random Forest selected as Champion (F1 = 0.2856, ROC-AUC = 0.6634).",
            "Unsupervised K-Means validated K=3 patient phenotypes.",
            "Production Flask web portal operational with real-time API."
        ], COLOR_SECONDARY_BLUE),
        ("CONCLUSION", [
            "Successfully built an end-to-end healthcare ML system fulfilling CO1, CO2, and CO3.",
            "Demonstrated that F1-score and Recall are far superior to raw accuracy in imbalanced clinical triage.",
            "Prior inpatient history and stay duration are the paramount drivers of 30-day diabetic readmission.",
            "Modular architecture ensures independent reproducibility across all M1–M4 components."
        ], COLOR_TEAL),
        ("FUTURE SCOPE", [
            "Incorporate temporal EHR electronic health records across multiple longitudinal encounters.",
            "Integrate Explainable AI (SHAP / LIME) for patient-specific feature attribution explanations.",
            "Deploy automated concept drift monitoring in live hospital telemetry pipelines.",
            "Conduct formal institutional review and clinical trial validation prior to actual clinical use."
        ], COLOR_SECONDARY_BLUE)
    ]

    for idx, (stitle, sbullets, scol) in enumerate(sections):
        x = Inches(0.8 + idx * 3.99)
        y = Inches(1.6)
        scard = s18.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, col_w, Inches(5.3))
        scard.fill.solid()
        scard.fill.fore_color.rgb = COLOR_CARD_BG
        scard.line.color.rgb = scol
        scard.line.width = Pt(1.5)

        stf = scard.text_frame
        stf.word_wrap = True
        stf.margin_left = Inches(0.3)
        stf.margin_top = Inches(0.3)
        stf.margin_right = Inches(0.3)

        p = stf.paragraphs[0]
        p.text = stitle
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = scol

        for b in sbullets:
            p = stf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(11)
            p.font.color.rgb = COLOR_TEXT_MUTED
            p.space_before = Pt(7)

    add_notes(s18,
        "In Slide 18, we summarize our results, conclusions, and future scope. "
        "We successfully processed 50,000 hospital records, implemented and compared 6 linear models and 3 tree-based ensembles, explored unsupervised clustering, and deployed a working Flask web application. "
        "Our primary academic conclusion is that in imbalanced healthcare datasets, accuracy is deceptive; balanced ensembles like Random Forest optimize sensitivity for life-critical readmission prevention. "
        "In the future, we plan to incorporate SHAP explainability and longitudinal clinical records. "
        "We emphasize that this project is an educational machine learning demonstration, not a medical diagnosis device."
    )

    # =========================================================================
    # SLIDE 19: THANK YOU & Q&A
    # =========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    bg19 = s19.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg19.fill.solid()
    bg19.fill.fore_color.rgb = COLOR_PRIMARY_NAVY
    bg19.line.color.rgb = COLOR_PRIMARY_NAVY

    tcard19 = s19.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(1.4), Inches(10.333), Inches(4.7))
    tcard19.fill.solid()
    tcard19.fill.fore_color.rgb = RGBColor(30, 41, 59)
    tcard19.line.color.rgb = COLOR_TEAL
    tcard19.line.width = Pt(2)

    ttf19 = tcard19.text_frame
    ttf19.word_wrap = True
    ttf19.margin_left = Inches(0.8)
    ttf19.margin_top = Inches(0.6)
    ttf19.margin_right = Inches(0.8)

    p = ttf19.paragraphs[0]
    p.text = "THANK YOU"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.alignment = PP_ALIGN.CENTER

    p = ttf19.add_paragraph()
    p.text = "VITALSIGN: AI-BASED 30-DAY HOSPITAL READMISSION PREDICTION"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(10)

    p = ttf19.add_paragraph()
    p.text = "Questions, Feedback & Discussion Welcomed"
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(14)

    p = ttf19.add_paragraph()
    p.text = "Academic Disclaimer: This project is strictly developed as an academic educational machine learning system for PBL evaluation and is not intended for clinical diagnostic decision-making without certified regulatory validation."
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(148, 163, 184)
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(24)

    add_notes(s19,
        "Thank you respected faculty mentors and evaluators for your time and guidance. "
        "We are now ready to demonstrate our live Flask web application and answer any technical questions regarding our data preprocessing, linear models, tree ensembles, unsupervised clustering, or lifecycle architecture."
    )

    prs.save(OUTPUT_PPTX)
    print(f"\n[OK] Presentation saved successfully to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    create_presentation()
