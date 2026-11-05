"""
streamlit_app.py - Main Entrypoint for Resume Job Compatibility Analyzer

Academic ML Lab Live Demonstration Application
Built with Streamlit, Scikit-learn, Pandas, and Matplotlib.
Supports interactive resume analysis (PDF/DOCX/TXT), multi-model benchmarking,
skill gap diagnosis, and publication-quality diagnostic performance dashboards.

Design: Minimal, modern B2B SaaS analytics theme (Linear / Stripe inspired).
Strictly zero emojis, professional Lucide SVG icons, deep muted green accents.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Setup Path Resolution
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
from resume_parser import parse_resume_file, estimate_years_of_experience
from analyzer import (
    analyze_compatibility,
    load_ml_pipeline_and_models,
    VALID_CATEGORIES,
    CATEGORY_DISPLAY_MAP
)
from components import (
    apply_custom_styles,
    render_header,
    render_compatibility_badge,
    render_metric_card,
    render_workflow_step,
    render_skill_chips,
    render_academic_disclaimer,
    get_icon
)


# Set Streamlit Page Configuration (Clean title, standard favicon)
st.set_page_config(
    page_title="Resume Job Compatibility Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply unified minimal styling
apply_custom_styles()


# Sample presets for quick live professor demonstration
DEMO_PRESETS = {
    "Select a Demo Preset...": {
        "title": "",
        "job_desc": "",
        "resume_text": "",
        "job_cat": "INFORMATION-TECHNOLOGY",
        "cand_cat": "INFORMATION-TECHNOLOGY",
        "job_exp": 5.0,
        "cand_exp": 6.0,
    },
    "High Match: Senior Machine Learning Engineer (IT)": {
        "title": "Senior Machine Learning Engineer",
        "job_desc": """We are seeking an experienced Senior Machine Learning Engineer to design, build, and deploy production ML models.
Requirements:
- 5+ years of experience in Python, Scikit-learn, Pandas, NumPy, and SQL.
- Deep expertise in classical machine learning algorithms, model evaluation, and feature engineering.
- Experience building REST APIs with FastAPI or Flask and containerizing with Docker.
- Experience with CI/CD pipelines, Git, and Linux environments.
- Strong background in data modeling and statistics.""",
        "resume_text": """Alex Morgan - Senior Machine Learning Engineer
Summary: Results-driven ML Engineer with 6.5 years of industry experience developing classical and applied ML solutions.
Technical Skills:
- Languages & Frameworks: Python, Java, SQL, Scikit-learn, Pandas, NumPy, SciPy, Matplotlib.
- Web & APIs: FastAPI, Flask, REST APIs, JSON, Postman.
- DevOps & Cloud: Docker, Kubernetes, Git, CI/CD, Linux, AWS, Bash.
- Databases: PostgreSQL, MySQL, Redis, MongoDB.
Work Experience:
Lead Data Scientist at TechCorp (2021 - Present): Architected end-to-end ML prediction pipelines. Optimized feature selection using ANOVA and mutual information. Containerized models using Docker.
ML Developer at DataSoft (2018 - 2021): Implemented classical classifiers (Random Forest, SVM, KNN, Logistic Regression) with cross-validation.
Education: B.Tech in Computer Science & Engineering.""",
        "job_cat": "INFORMATION-TECHNOLOGY",
        "cand_cat": "INFORMATION-TECHNOLOGY",
        "job_exp": 5.0,
        "cand_exp": 6.5,
    },
    "Moderate Match: Junior Business Analyst (Finance)": {
        "title": "Financial Business Analyst",
        "job_desc": """Looking for a Financial Business Analyst to join our quantitative research group.
Responsibilities:
- Conduct financial modeling, variance analysis, and budget forecasting.
- Strong proficiency in Advanced Excel, SQL, Tableau, and Financial Analysis.
- 4+ years experience in corporate finance, budgeting, and ERP systems (SAP).
- Experience working with cross-functional teams and executive leadership.""",
        "resume_text": """Priya Sharma - Financial Associate
Professional Experience: 3 years of corporate accounting and financial data analysis.
Skills: Excel, Advanced Excel, Financial Analysis, Accounting, Budgeting, SQL, PowerPoint.
Experience:
Financial Analyst at Global Corp (2022 - Present): Maintained monthly balance sheets and created revenue forecast sheets in Excel. Queried transactional data with SQL.
Junior Accountant at FinTax (2020 - 2022): Prepared tax returns and reconciled bank accounts.
Education: Bachelor of Commerce in Accounting.""",
        "job_cat": "FINANCE",
        "cand_cat": "FINANCE",
        "job_exp": 4.0,
        "cand_exp": 3.0,
    },
    "Low / Weak Match: HR Manager applying for Software Architect (IT)": {
        "title": "Principal Cloud Software Architect",
        "job_desc": """Seeking a Principal Software Architect to lead enterprise cloud migrations.
Required Skills:
- 8+ years experience in Java, Spring Boot, Microservices, Kubernetes, Docker, and AWS.
- Deep expertise in distributed systems, REST APIs, PostgreSQL, Kafka, and System Design.
- Proven leadership in software engineering architecture.""",
        "resume_text": """David Miller - Senior Human Resources Manager
Summary: HR Professional with 8 years experience leading talent acquisition, employee engagement, and organizational development.
Skills: Recruitment, Talent Acquisition, Payroll, HRIS, Employee Relations, Conflict Resolution, Leadership, Communication.
Experience:
HR Manager at PeopleFirst (2019 - Present): Supervised hiring for technical teams and administered company benefits.
HR Specialist (2016 - 2019): Managed onboarding and compliance.
Education: Master of Business Administration in Human Resources.""",
        "job_cat": "INFORMATION-TECHNOLOGY",
        "cand_cat": "HR",
        "job_exp": 8.0,
        "cand_exp": 8.0,
    }
}


def render_sidebar_navigation() -> str:
    """Renders minimal, professional sidebar navigation with branding and Muskan footer."""
    with st.sidebar:
        # Minimal brand header
        brain_icon = get_icon("brain-circuit", size=18, color="#245B4A")
        st.markdown(f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">{brain_icon}</div>
            <div>
                <div class="sidebar-brand-title">Resume Analyzer</div>
                <div class="sidebar-brand-sub">ML Lab Mini-Project</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Resume Analyzer",
                "Model Performance",
                "About Project"
            ],
            index=1,
            label_visibility="collapsed"
        )

        # Subtle footer pinned at bottom of sidebar
        user_icon = get_icon("user", size=14, color="#245B4A")
        st.markdown(f"""
        <div class="sidebar-footer">
            <div class="sidebar-footer-content">
                {user_icon}
                <span>Created by Muskan</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return page


# ==============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==============================================================================
def render_dashboard_overview():
    # 1. Dashboard Header
    render_header(
        title="Project Overview",
        subtitle="Resume Job Compatibility Analyzer | ML Lab Mini-Project",
        tag="ML Lab Mini-Project",
        icon_name="layout-dashboard"
    )

    st.markdown("""
    <p style="color: #475569; font-size: 0.92rem; line-height: 1.55; margin: -0.5rem 0 1.25rem 0;">
    An academic machine learning application that evaluates compatibility between candidate resumes and job descriptions using text processing, feature engineering and classical machine learning algorithms.
    </p>
    """, unsafe_allow_html=True)

    # 2. Project Statistics (4 compact metric cards with verified dataset values)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Dataset Records", "122,560", "Resume-Job Match Pairs", icon_name="database")
    with c2:
        render_metric_card("Candidate Profiles", "9,532", "Unique Candidates", icon_name="file-text")
    with c3:
        render_metric_card("Job Postings", "1,164", "Unique Jobs in the merged dataset", icon_name="briefcase-business")
    with c4:
        render_metric_card("Best Validation Macro F1", "0.7269", "KNN (Primary Classifier)", icon_name="chart-no-axes-combined")

    st.markdown("---")

    # 3. Project Workflow Pipeline (6 clear steps)
    pipe_icon = get_icon("layers", size=16, color="#245B4A")
    st.markdown(f"#### {pipe_icon} Project Workflow Pipeline", unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    with w1:
        render_workflow_step(
            1,
            "Dataset Collection",
            "Resume dataset, job dataset and match records.",
            icon_name="database"
        )
    with w2:
        render_workflow_step(
            2,
            "Data Preprocessing",
            "Data cleaning, merging, duplicate and missing-value checks. Target-derived leakage columns excluded from model inputs.",
            icon_name="filter"
        )
    with w3:
        render_workflow_step(
            3,
            "Feature Engineering",
            "TF-IDF with unigrams and bigrams (1,500 text features). 16 structured features. Cosine similarity and Jaccard similarity.",
            icon_name="sliders"
        )

    st.write("")

    w4, w5, w6 = st.columns(3)
    with w4:
        render_workflow_step(
            4,
            "Model Training",
            "KNN, Linear SVM, Logistic Regression, Random Forest, Naive Bayes.",
            icon_name="brain-circuit"
        )
    with w5:
        render_workflow_step(
            5,
            "Model Evaluation",
            "Accuracy, Precision, Recall and Macro F1. Candidate-grouped train, validation and test split.",
            icon_name="chart-no-axes-combined"
        )
    with w6:
        render_workflow_step(
            6,
            "Prediction",
            "Weak Positive, Medium Positive, Strong Positive.",
            icon_name="circle-check"
        )

    st.markdown("---")

    # 5. Core Algorithms Used & 6. Validation Methodology (Side by Side)
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        alg_icon = get_icon("brain-circuit", size=16, color="#245B4A")
        st.markdown(f"#### {alg_icon} Core Machine Learning Models", unsafe_allow_html=True)
        st.markdown("""
        The system evaluates compatibility using five classical machine learning algorithms trained on genuine text and legitimate structured features:
        """)

        st.markdown("""
| Model | Feature Representation | Validation Macro F1 | Role in System |
| :--- | :--- | :---: | :--- |
| **K-Nearest Neighbors (KNN)** | Structured Features (16 dims) | **0.7269** | **Primary Classifier** (Highest Val Macro F1) |
| **Support Vector Machine (Linear SVM)** | All Features (1,516 dims) | 0.7082 | Secondary Classifier (LinearSVC Hyperplane) |
| **Random Forest** | Structured Features (16 dims) | 0.6759 | Ensemble Classifier (100 Trees) |
| **Logistic Regression** | All Features (1,516 dims) | 0.6395 | Linear Baseline (Softmax with L2) |
| **Naive Bayes** | TF-IDF Text (1,500 dims) | 0.4548 | Probabilistic Text Baseline (MultinomialNB) |
        """)

        check_icon = get_icon("circle-check", size=14, color="#245B4A")
        st.markdown(f"""
        <div style="background-color: #EAF3EF; border: 1px solid #D1E5DC; border-radius: 6px; padding: 0.75rem 0.9rem; margin-top: 0.75rem; font-size: 0.82rem; color: #172B25; line-height: 1.45;">
            <b>{check_icon} Primary Model Selection:</b> <b>K-Nearest Neighbors (KNN)</b> is designated as the primary classification engine because it achieved the highest validation Macro F1 score (<b>0.7269</b>) under candidate-grouped evaluation, demonstrating superior instance-level discrimination across compatibility categories.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        val_icon = get_icon("layers", size=16, color="#245B4A")
        st.markdown(f"#### {val_icon} Validation Methodology", unsafe_allow_html=True)
        st.markdown("""
        To evaluate real-world generalization and prevent candidate identity leakage, the dataset is partitioned using candidate-grouped splitting:
        """)

        st.markdown("""
| Split Partition | Record Count | Percentage | Candidate Overlap |
| :--- | :---: | :---: | :---: |
| **Training Set** | 85,392 | 69.67% | Zero (Disjoint) |
| **Validation Set** | 18,484 | 15.08% | Zero (Disjoint) |
| **Held-Out Test Set** | 18,684 | 15.24% | Zero (Disjoint) |
| **Total Merged** | **122,560** | **100.00%** | **Disjoint Split (~70:15:15)** |
        """)

        shield_icon = get_icon("activity", size=14, color="#245B4A")
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid #245B4A; border-radius: 6px; padding: 0.75rem 0.9rem; margin-top: 0.75rem; font-size: 0.82rem; color: #475569; line-height: 1.45;">
            <b>{shield_icon} Leakage Prevention Guardrail:</b> Candidate-grouped splitting ensures that all match pairs for any given candidate exist strictly within a single partition (zero candidate overlap across splits). This prevents candidate identity memorization and guarantees that validation and test evaluations measure generalization to unseen candidate profiles.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Dataset Information
    data_icon = get_icon("database", size=16, color="#245B4A")
    st.markdown(f"#### {data_icon} Dataset Information", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([1.1, 0.9])

    with col_d1:
        st.markdown("""
        The project is trained and evaluated on the public benchmark dataset from Hugging Face:
        """)
        st.markdown("""
| File Name | Original Records | Description / Scope |
| :--- | :---: | :--- |
| `cv.csv` | 9,544 records | Candidate profiles with resumes, skills, target position, experience, education |
| `job.csv` | 1,167 records | Job postings with job titles, descriptions, required skills, and experience |
| `matches.csv` | 122,560 records | Pairwise candidate-job matches with compatibility ground truth |
        """)

    with col_d2:
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 0.5rem;">
            <div class="metric-title">{get_icon("database", size=13, color="#64748B")} Dataset Repository Source</div>
            <div style="font-size: 0.92rem; font-weight: 600; color: #172B25; margin: 0.35rem 0;">
                Hugging Face: <a href="https://huggingface.co/datasets/nonameee12233/job-resume-matching" target="_blank" style="color: #245B4A; text-decoration: underline;">nonameee12233/job-resume-matching</a>
            </div>
            <div class="metric-subtitle">
                Merged via <code>candidate_id</code> and <code>job_id</code> into <b>122,560</b> total pairwise match records covering <b>9,532</b> unique candidates and <b>1,164</b> unique jobs.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Mandatory Academic Dataset Note
    info_icon = get_icon("info", size=15, color="#245B4A")
    st.markdown(f"""
    <div class="academic-disclaimer" style="margin-top: 0.5rem;">
        <div class="disclaimer-title">{info_icon} Academic Dataset Notice</div>
        The classification labels are derived from the dataset's existing compatibility score. The model learns the dataset's scoring patterns and does not predict actual hiring decisions.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: RESUME ANALYZER (LIVE INTERACTIVE TOOL)
# ==============================================================================
def render_resume_analyzer():
    render_header(subtitle="Live Multi-Model Inference &bull; Skill Gap Diagnosis &bull; Document Text Ingestion")

    st.markdown("### Live Compatibility Analysis")
    st.markdown("Upload or paste candidate details and job requirements to evaluate compatibility across classical ML models.")

    # Demonstration Preset Selector (Clean, professional label)
    selected_preset = st.selectbox(
        "Quick Demo Presets (Select to auto-populate fields for testing):",
        list(DEMO_PRESETS.keys()),
        index=0
    )

    preset_data = DEMO_PRESETS[selected_preset]

    # Two-Column Input Form
    col_resume, col_job = st.columns(2)

    with col_resume:
        file_icon = get_icon("file-text", size=16, color="#245B4A")
        st.markdown(f"#### {file_icon} 1. Candidate Resume", unsafe_allow_html=True)
        input_mode = st.radio("Resume Input Mode:", ["Upload Document (PDF / DOCX / TXT)", "Paste Resume Text"], horizontal=True)

        resume_text = ""
        uploaded_doc = None

        if input_mode == "Upload Document (PDF / DOCX / TXT)":
            uploaded_doc = st.file_uploader(
                "Choose Resume File:",
                type=["pdf", "docx", "txt"],
                help="Accepts selectable PDF files, modern Microsoft Word (.docx), or plain text (.txt)."
            )
            if uploaded_doc is not None:
                with st.spinner("Extracting and parsing document text..."):
                    parse_result = parse_resume_file(uploaded_doc)
                    if parse_result["success"]:
                        resume_text = parse_result["text"]
                        check_icon = get_icon("circle-check", size=14, color="#245B4A")
                        st.markdown(f"<span style='color: #245B4A; font-size: 0.82rem;'>{check_icon} Parsed <b>{parse_result['filename']}</b> ({parse_result['word_count']} words)</span>", unsafe_allow_html=True)
                    else:
                        st.error(f"Error parsing document: {parse_result['error']}")
        else:
            resume_text = st.text_area(
                "Paste Resume Body:",
                value=preset_data["resume_text"],
                height=220,
                placeholder="Paste extracted resume text, skills, and work experience here..."
            )

        # Fallback to preset text if uploaded_doc is None and preset selected
        if not resume_text and preset_data["resume_text"]:
            resume_text = preset_data["resume_text"]

        # Structured candidate inputs
        c_cat, c_exp = st.columns(2)
        with c_cat:
            default_cand_idx = VALID_CATEGORIES.index(preset_data["cand_cat"]) if preset_data["cand_cat"] in VALID_CATEGORIES else 0
            candidate_category = st.selectbox(
                "Candidate Professional Domain:",
                VALID_CATEGORIES,
                index=default_cand_idx,
                format_func=lambda x: CATEGORY_DISPLAY_MAP.get(x, x)
            )

        with c_exp:
            # Auto-estimate experience if text available
            heuristic_exp = estimate_years_of_experience(resume_text) if resume_text else preset_data["cand_exp"]
            cand_exp_val = preset_data["cand_exp"] if selected_preset != "Select a Demo Preset..." else heuristic_exp
            candidate_experience = st.number_input(
                "Total Work Experience (Years):",
                min_value=0.0,
                max_value=40.0,
                value=float(cand_exp_val),
                step=0.5
            )

        with st.expander("Inspect Raw Extracted Resume Text"):
            if resume_text:
                st.text_area("Extracted Body:", resume_text, height=120, disabled=True)
                st.caption(f"Word count: {len(resume_text.split())} words | Characters: {len(resume_text)}")
            else:
                st.info("No resume text loaded yet.")

    with col_job:
        job_icon = get_icon("briefcase-business", size=16, color="#245B4A")
        st.markdown(f"#### {job_icon} 2. Job Description & Requirements", unsafe_allow_html=True)
        job_title = st.text_input(
            "Target Job Title:",
            value=preset_data["title"],
            placeholder="e.g. Senior Machine Learning Engineer"
        )

        job_description = st.text_area(
            "Job Description Text:",
            value=preset_data["job_desc"],
            height=180,
            placeholder="Paste employer job description, responsibilities, and required competencies here..."
        )

        j_cat, j_exp = st.columns(2)
        with j_cat:
            default_job_idx = VALID_CATEGORIES.index(preset_data["job_cat"]) if preset_data["job_cat"] in VALID_CATEGORIES else 0
            job_category = st.selectbox(
                "Job Functional Category:",
                VALID_CATEGORIES,
                index=default_job_idx,
                format_func=lambda x: CATEGORY_DISPLAY_MAP.get(x, x)
            )

        with j_exp:
            job_experience_required = st.number_input(
                "Required Experience (Years):",
                min_value=0.0,
                max_value=40.0,
                value=float(preset_data["job_exp"]),
                step=0.5
            )

        st.caption("Provide clear responsibilities and technical keywords for maximum alignment accuracy.")

    # Action Button (Clean, deep muted green styling)
    st.markdown("---")
    analyze_btn = st.button("Run Compatibility Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        if not resume_text.strip():
            st.error("Please provide a resume by uploading a document or pasting text.")
            return
        if not job_description.strip():
            st.error("Please enter the job description text.")
            return

        with st.spinner("Executing feature transformation and multi-model inference..."):
            results = analyze_compatibility(
                resume_text=resume_text,
                job_description=job_description,
                years_experience=candidate_experience,
                years_required=job_experience_required,
                candidate_category=candidate_category,
                job_category=job_category
            )

        if not results.get("success"):
            st.error(f"Analysis Failed: {results.get('error')}")
            return

        # ----------------------------------------------------------------------
        # DISPLAY RESULTS DASHBOARD
        # ----------------------------------------------------------------------
        chart_icon = get_icon("chart-no-axes-combined", size=18, color="#245B4A")
        st.markdown(f"### {chart_icon} Compatibility Assessment Results", unsafe_allow_html=True)

        primary_class = results["primary_prediction"]
        consensus_class = results["consensus_prediction"]
        confidence = results.get("primary_confidence")

        # Top Card: Primary Model Decision
        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 3px solid #245B4A;">
                <div class="metric-title">{get_icon("brain-circuit", size=14, color="#245B4A")} Primary Model Prediction (K-Nearest Neighbors)</div>
                <div style="margin: 0.6rem 0;">{render_compatibility_badge(primary_class)}</div>
                <div class="metric-subtitle">
                    {f"<b>Estimated Confidence:</b> {confidence*100:.1f}%" if confidence is not None else "Confidence: Calibrated Decision"}
                    <br><b>Multi-Model Consensus:</b> {consensus_class.replace('_', ' ').title()} ({results['model_agreement_count']}/{results['total_models']} Models Agree)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            gap = results["skill_gap"]
            # 4 KPI Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                cov = gap["requirement_coverage"] * 100
                render_metric_card("Skill Coverage", f"{cov:.0f}%", f"{len(gap['matched_skills'])} of {gap['total_job_skills']} Req", icon_name="circle-check")
            with kpi2:
                jac = gap["jaccard_similarity"] * 100
                render_metric_card("Jaccard Overlap", f"{jac:.1f}%", "Set Overlap", icon_name="layers")
            with kpi3:
                exp_diff = results["experience_gap"]
                sign = "+" if exp_diff > 0 else ""
                render_metric_card("Experience Gap", f"{sign}{exp_diff:.1f}y", f"{results['years_experience']}y vs {results['years_required']}y", icon_name="clock")
            with kpi4:
                sim = results["tfidf_similarity"] * 100
                render_metric_card("Text Similarity", f"{sim:.1f}%", "TF-IDF Cosine", icon_name="file-search")

        st.markdown("---")

        # Skill Breakdown
        st.markdown(f"#### {get_icon('file-search', size=16, color='#245B4A')} Skill Gap Analysis", unsafe_allow_html=True)
        sk_col1, sk_col2, sk_col3 = st.columns(3)

        with sk_col1:
            st.markdown(f"**Matched Competencies ({len(gap['matched_skills'])}):**")
            render_skill_chips(gap["matched_skills"], chip_type="matched")

        with sk_col2:
            st.markdown(f"**Missing Required Skills ({len(gap['missing_skills'])}):**")
            render_skill_chips(gap["missing_skills"], chip_type="missing")

        with sk_col3:
            st.markdown(f"**Additional Candidate Skills ({len(gap['additional_skills'])}):**")
            render_skill_chips(gap["additional_skills"], chip_type="additional")

        st.markdown("---")

        # Multi-Model Comparison Table
        st.markdown(f"#### {get_icon('brain-circuit', size=16, color='#245B4A')} Multi-Model Comparative Verdicts", unsafe_allow_html=True)
        st.markdown("All models evaluated on identical transformed features without refitting the preprocessing pipeline.")

        model_rows = []
        for model_name, m_res in results["predictions"].items():
            conf_str = f"{m_res['confidence']*100:.1f}%" if m_res.get("confidence") is not None else "N/A (Geometric Hyperplane)"
            proba_details = ""
            if m_res.get("probabilities"):
                p = m_res["probabilities"]
                proba_details = f"Weak: {p.get('weak_positive', 0)*100:.1f}%, Med: {p.get('medium_positive', 0)*100:.1f}%, Str: {p.get('strong_positive', 0)*100:.1f}%"
            else:
                proba_details = "Direct class decision (predict_proba not applicable)"

            model_rows.append({
                "Model Architecture": model_name + (" (Primary)" if m_res["is_primary"] else ""),
                "Feature Representation": m_res["features_used"],
                "Predicted Category": m_res["prediction"].replace("_", " ").title(),
                "Confidence / Top Probability": conf_str,
                "Class Distribution Breakdown": proba_details
            })

        df_model_comparison = pd.DataFrame(model_rows)
        st.dataframe(df_model_comparison, use_container_width=True, hide_index=True)

        render_academic_disclaimer()


# ==============================================================================
# PAGE 3: MODEL PERFORMANCE BENCHMARK
# ==============================================================================
def render_model_performance():
    render_header(subtitle="Empirical Benchmarks &bull; Confusion Matrices &bull; Diagnostic Visualizations")

    st.markdown("### Model Evaluation Results & Benchmark")
    st.markdown("Loaded directly from verified experiment executions (`outputs/results/`).")

    # Load Saved Classification Comparison CSV
    clf_csv_path = RESULTS_DIR / "classification_model_comparison.csv"
    if clf_csv_path.exists():
        df_clf = pd.read_csv(clf_csv_path)

        st.markdown("#### Classical Classifier Benchmark (Ranked by Validation Macro F1)")
        st.markdown("""
        *Methodological Guardrail:* Models are ranked strictly by **Validation Macro F1** to prevent test-set tuning bias. 
        Test set metrics represent the final single-pass evaluation on candidate-grouped held-out test data.
        """)

        # Display table with formatting
        st.dataframe(
            df_clf.style.highlight_max(subset=["Val_Macro_F1", "Test_Macro_F1"], color="#EAF3EF"),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error(f"Classification benchmark file not found at `{clf_csv_path}`")

    st.markdown("---")

    # Visual Diagnostics Tabs
    st.markdown("#### Diagnostic Plots & Laboratory Figures")
    tab_cm, tab_bpnn, tab_tree, tab_pca, tab_kmeans, tab_feat = st.tabs([
        "Confusion Matrices",
        "Backprop Neural Net",
        "ID3 Decision Tree",
        "PCA / TruncatedSVD",
        "K-Means Clustering",
        "Feature Selection"
    ])

    with tab_cm:
        cm_plot = PLOTS_DIR / "confusion_matrices_classification.png"
        if cm_plot.exists():
            st.image(str(cm_plot), caption="Normalized Confusion Matrices across Classifiers", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found.")

    with tab_bpnn:
        bpnn_plot = PLOTS_DIR / "bpnn_loss_and_cm.png"
        if bpnn_plot.exists():
            st.image(str(bpnn_plot), caption="Experiment 5: Multi-Layer Perceptron Loss Curve & Validation Confusion Matrix", use_container_width=True)
            bpnn_csv = RESULTS_DIR / "bpnn_mlp_metrics.csv"
            if bpnn_csv.exists():
                st.dataframe(pd.read_csv(bpnn_csv), use_container_width=True, hide_index=True)
        else:
            st.info("BPNN plot not found.")

    with tab_tree:
        tree_plot = PLOTS_DIR / "id3_decision_tree.png"
        imp_plot = PLOTS_DIR / "id3_feature_importance.png"
        c1, c2 = st.columns([1.4, 0.8])
        with c1:
            if tree_plot.exists():
                st.image(str(tree_plot), caption="Experiment 7: Entropy-based Decision Tree (Max Depth = 4)", use_container_width=True)
        with c2:
            if imp_plot.exists():
                st.image(str(imp_plot), caption="Shannon Information Gain Feature Importances", use_container_width=True)
                imp_csv = RESULTS_DIR / "id3_feature_importances.csv"
                if imp_csv.exists():
                    st.dataframe(pd.read_csv(imp_csv).head(6), use_container_width=True, hide_index=True)

    with tab_pca:
        scree_plot = PLOTS_DIR / "pca_svd_scree.png"
        scatter_plot = PLOTS_DIR / "pca_struct_scatter.png"
        c1, c2 = st.columns(2)
        with c1:
            if scree_plot.exists():
                st.image(str(scree_plot), caption="Experiment 6: TruncatedSVD (LSA) Scree & Cumulative Variance", use_container_width=True)
        with c2:
            if scatter_plot.exists():
                st.image(str(scatter_plot), caption="2D PCA Projection of Structured Features", use_container_width=True)

        pca_csv = RESULTS_DIR / "pca_svd_summary.csv"
        if pca_csv.exists():
            st.dataframe(pd.read_csv(pca_csv), use_container_width=True, hide_index=True)

    with tab_kmeans:
        elbow_plot = PLOTS_DIR / "kmeans_elbow_analysis.png"
        proj_plot = PLOTS_DIR / "kmeans_cluster_projection.png"
        c1, c2 = st.columns(2)
        with c1:
            if elbow_plot.exists():
                st.image(str(elbow_plot), caption="Experiment 11: Elbow Method (Inertia) & Silhouette Analysis", use_container_width=True)
        with c2:
            if proj_plot.exists():
                st.image(str(proj_plot), caption="2D PCA Projection of K-Means Discovered Clusters (k=3)", use_container_width=True)

        km_contingency = RESULTS_DIR / "kmeans_contingency_table.csv"
        if km_contingency.exists():
            st.markdown("**Contingency Cross-Tabulation (Unsupervised Clusters vs Supervised Labels):**")
            st.dataframe(pd.read_csv(km_contingency), use_container_width=True)

    with tab_feat:
        feat_plot = PLOTS_DIR / "feature_selection_importance.png"
        if feat_plot.exists():
            st.image(str(feat_plot), caption="Experiment 1: Top Features Selected by ANOVA F-test (SelectKBest, k=500)", use_container_width=True)
        feat_csv = RESULTS_DIR / "feature_selection_summary.csv"
        if feat_csv.exists():
            st.dataframe(pd.read_csv(feat_csv), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Regression Table
    reg_csv_path = RESULTS_DIR / "regression_model_comparison.csv"
    if reg_csv_path.exists():
        st.markdown("#### Continuous Final Score Prediction (Linear vs Ridge Regression)")
        st.dataframe(pd.read_csv(reg_csv_path), use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 4: ABOUT PROJECT & METHODOLOGY
# ==============================================================================
def render_about_project():
    render_header(subtitle="Academic Rigor &bull; Data Integrity &bull; Methodology Safeguards")

    st.markdown("### About the Machine Learning Lab Mini-Project")

    st.markdown("""
    #### Educational Objectives
    1. **Mastery of Classical ML Algorithms:** Provide rigorous empirical comparisons across 11 core algorithms studied in the university ML curriculum.
    2. **Prevention of Target & Group Leakage:** Demonstrate industry-grade research methodology by enforcing candidate-grouped partitioning and quarantining synthetic sub-scores.
    3. **Algorithmic Interpretability:** Avoid 'black box' claims by examining decision boundaries, Shannon entropy gains, principal component projections, and feature weights.
    4. **CPU-Friendly Engineering:** Build scalable, robust ML pipelines capable of running on standard student laptop hardware in seconds.

    ---

    #### Data Leakage Audit Findings
    During Phase 2 Step 1, a statistical leakage audit was performed on the Hugging Face dataset `nonameee12233/job-resume-matching`:
    - **Discovery 1:** `raw_score` is an exact linear combination ($R^2 = 1.0$) of intermediate sub-scores.
    - **Discovery 2:** Ground-truth classification `label_strength` is a strict deterministic binning of `final_score` at cutoffs $0.58$ and $0.68$.
    - **Safety Protocol:** All precomputed sub-scores (`semantic_score`, `skill_score`, `position_score`, `category_score`, `experience_score`, `education_score`, `candidate_rank`) and continuous targets were quarantined and excluded from model inputs.

    ---

    #### Candidate-Grouped Train / Validation / Test Splitting
    - **Total Records:** 122,560 match pairs across 9,544 unique candidates.
    - **Splitting Strategy:** Grouped strictly by `candidate_id` into 70% Train (85,392 rows), 15% Validation (18,484 rows), and 15% Held-out Test (18,684 rows).
    - **Limitation & Rationale:** Prevents identity memorization while balancing category distributions across folds.

    ---

    #### Technology Stack
    - **Languages & Frameworks:** Python 3.10+, Scikit-learn, SciPy, NumPy, Pandas, Matplotlib, Streamlit.
    - **Document Parsing:** `pypdf` (PDF extraction) and `python-docx` (DOCX extraction).
    - **Hardware:** Local Windows CPU Only (Zero GPU required).
    """)

    render_academic_disclaimer()


# ==============================================================================
# MAIN ROUTING CONTROLLER
# ==============================================================================
def main():
    selected_page = render_sidebar_navigation()

    if selected_page == "Dashboard":
        render_dashboard_overview()
    elif selected_page == "Resume Analyzer":
        render_resume_analyzer()
    elif selected_page == "Model Performance":
        render_model_performance()
    elif selected_page == "About Project":
        render_about_project()


if __name__ == "__main__":
    main()
