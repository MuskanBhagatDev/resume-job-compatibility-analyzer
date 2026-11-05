"""
analyzer.py - Core ML Inference, Compatibility Scoring & Skill Gap Analysis Engine

Loads pre-trained classical ML models and the fitted CompatibilityFeaturePipeline.
Transforms input resume-job pairs with ZERO pipeline re-fitting to guarantee 100%
consistency with the training distribution. Computes transparent skill gap metrics,
text similarity, and multi-model consensus predictions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Setup import paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import feature_engineering
from resume_parser import extract_skills_from_text


# Category list matching the training pipeline
VALID_CATEGORIES = [
    "INFORMATION-TECHNOLOGY",
    "FINANCE",
    "SALES",
    "BUSINESS-DEVELOPMENT",
    "HR"
]

CATEGORY_DISPLAY_MAP = {
    "INFORMATION-TECHNOLOGY": "Information Technology (IT)",
    "FINANCE": "Finance & Accounting",
    "SALES": "Sales & Retail",
    "BUSINESS-DEVELOPMENT": "Business Development & Strategy",
    "HR": "Human Resources (HR)"
}


@st.cache_resource
def load_ml_pipeline_and_models() -> Dict[str, Any]:
    """
    Loads saved feature pipeline and model artifacts from models/ directory.
    Cached via Streamlit to ensure instant page reloads and minimal memory overhead.
    """
    models_dir = config.MODELS_DIR
    artifacts: Dict[str, Any] = {
        "pipeline": None,
        "models": {},
        "errors": []
    }

    # 1. Load fitted Feature Pipeline
    pipe_path = models_dir / "feature_pipeline.joblib"
    if pipe_path.exists():
        try:
            artifacts["pipeline"] = feature_engineering.CompatibilityFeaturePipeline.load(pipe_path)
        except Exception as e:
            artifacts["errors"].append(f"Failed to load feature pipeline: {str(e)}")
    else:
        artifacts["errors"].append(f"Feature pipeline file not found at {pipe_path}")

    # 2. Model definitions: (file_name, display_name, feature_representation)
    model_configs = [
        ("k_nearest_neighbors.joblib", "K-Nearest Neighbors (KNN)", "struct", True),
        ("support_vector_machine.joblib", "Support Vector Machine (Linear SVM)", "all", False),
        ("random_forest.joblib", "Random Forest Classifier", "struct", False),
        ("logistic_regression.joblib", "Logistic Regression", "all", False),
        ("bpnn_mlp.joblib", "Backprop Neural Network (MLP)", "struct", False),
    ]

    for filename, display_name, feat_type, is_primary in model_configs:
        m_path = models_dir / filename
        if m_path.exists():
            try:
                model_obj = joblib.load(m_path)
                artifacts["models"][display_name] = {
                    "model": model_obj,
                    "feat_type": feat_type,
                    "is_primary": is_primary,
                    "has_proba": hasattr(model_obj, "predict_proba"),
                    "classes": getattr(model_obj, "classes_", ["weak_positive", "medium_positive", "strong_positive"])
                }
            except Exception as e:
                artifacts["errors"].append(f"Failed loading {display_name}: {str(e)}")
        else:
            artifacts["errors"].append(f"Model file missing: {filename}")

    return artifacts


def perform_skill_gap_analysis(
    resume_skills: List[str],
    job_skills: List[str]
) -> Dict[str, Any]:
    """
    Computes transparent set-theoretic skill overlap metrics.
    
    Returns:
    - matched_skills: intersection (candidate possesses, job requires)
    - missing_skills: difference (job requires, candidate lacks)
    - additional_skills: difference (candidate possesses, not explicitly required)
    - jaccard_similarity: |A ∩ B| / |A ∪ B|
    - match_percentage: |A ∩ B| / |B| (Job requirement coverage)
    """
    set_cv = set([s.strip().lower() for s in resume_skills if s.strip()])
    set_job = set([s.strip().lower() for s in job_skills if s.strip()])

    matched = sorted(list(set_cv & set_job))
    missing = sorted(list(set_job - set_cv))
    additional = sorted(list(set_cv - set_job))

    union_len = len(set_cv | set_job)
    jaccard = round(len(matched) / union_len, 4) if union_len > 0 else 0.0
    req_coverage = round(len(matched) / len(set_job), 4) if len(set_job) > 0 else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_skills": additional,
        "jaccard_similarity": jaccard,
        "requirement_coverage": req_coverage,
        "total_cv_skills": len(set_cv),
        "total_job_skills": len(set_job)
    }


def analyze_compatibility(
    resume_text: str,
    job_description: str,
    years_experience: float,
    years_required: float,
    candidate_category: str,
    job_category: str,
    custom_resume_skills: Optional[List[str]] = None,
    custom_job_skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main evaluation pipeline for user input.
    Validates inputs, extracts skills, runs feature pipeline, and executes model inference.
    """
    if not resume_text.strip():
        return {"success": False, "error": "Resume text is empty. Please upload or paste a resume."}
    if not job_description.strip():
        return {"success": False, "error": "Job description is empty. Please provide job details."}

    # 1. Load artifacts
    artifacts = load_ml_pipeline_and_models()
    pipeline = artifacts.get("pipeline")
    models_dict = artifacts.get("models", {})

    if not pipeline:
        return {"success": False, "error": "ML Feature Pipeline is not available. Please ensure models are trained."}

    # 2. Skill Extraction & Gap Analysis
    extracted_cv_skills = extract_skills_from_text(resume_text, custom_resume_skills)
    extracted_job_skills = extract_skills_from_text(job_description, custom_job_skills)
    
    # Ensure comma-separated string format for pipeline
    cv_skills_str = ", ".join(extracted_cv_skills)
    job_skills_str = ", ".join(extracted_job_skills)

    skill_gap = perform_skill_gap_analysis(extracted_cv_skills, extracted_job_skills)

    # 3. Form single-row DataFrame for pipeline transformation
    sample_df = pd.DataFrame([{
        "candidate_id": "live_user_cand",
        "job_id": "live_user_job",
        "resume_text": resume_text,
        "job_description": job_description,
        "clean_skills_cv": cv_skills_str,
        "clean_skills_job": job_skills_str,
        "years_experience": float(years_experience),
        "years_required": float(years_required),
        "inferred_category": candidate_category.upper(),
        "job_category": job_category.upper()
    }])

    # 4. Transform features using fitted pipeline (NO REFITTING)
    try:
        features = pipeline.transform(sample_df)
        X_struct = features["X_struct"]  # Shape (1, 16)
        X_all = features["X_all"]        # Shape (1, 1516)
    except Exception as e:
        return {"success": False, "error": f"Feature transformation error: {str(e)}"}

    # Extract interaction metrics directly from unscaled calculation
    exp_gap = float(years_experience - years_required)
    
    # Retrieve unscaled tfidf similarity from pipeline computation
    tfidf_sim = float(pipeline._compute_tfidf_similarity(sample_df)[0])

    # 5. Execute Multi-Model Inference
    predictions = {}
    primary_result = None
    vote_counts: Dict[str, int] = {}

    for display_name, m_info in models_dict.items():
        model = m_info["model"]
        feat_type = m_info["feat_type"]
        X_input = X_struct if feat_type == "struct" else X_all

        try:
            pred_class = str(model.predict(X_input)[0])
            vote_counts[pred_class] = vote_counts.get(pred_class, 0) + 1

            proba_dict = None
            confidence = None

            if m_info["has_proba"]:
                try:
                    probas = model.predict_proba(X_input)[0]
                    classes = [str(c) for c in model.classes_]
                    proba_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probas)}
                    confidence = round(float(np.max(probas)), 4)
                except Exception:
                    proba_dict = None
                    confidence = None

            pred_entry = {
                "prediction": pred_class,
                "confidence": confidence,
                "probabilities": proba_dict,
                "features_used": "Structured Features (16 dims)" if feat_type == "struct" else "All Features (1,516 dims)",
                "is_primary": m_info["is_primary"],
                "has_proba": m_info["has_proba"]
            }

            predictions[display_name] = pred_entry
            if m_info["is_primary"]:
                primary_result = pred_entry

        except Exception as e:
            predictions[display_name] = {
                "prediction": "Error",
                "confidence": None,
                "probabilities": None,
                "error": str(e),
                "is_primary": m_info["is_primary"]
            }

    # If KNN missing, default primary to first valid
    if primary_result is None and predictions:
        first_key = list(predictions.keys())[0]
        primary_result = predictions[first_key]

    # Consensus prediction (majority vote)
    consensus_class = max(vote_counts.items(), key=lambda x: x[1])[0] if vote_counts else "Unknown"

    return {
        "success": True,
        "primary_prediction": primary_result["prediction"] if primary_result else "Unknown",
        "primary_confidence": primary_result.get("confidence") if primary_result else None,
        "primary_probabilities": primary_result.get("probabilities") if primary_result else None,
        "consensus_prediction": consensus_class,
        "model_agreement_count": vote_counts.get(consensus_class, 0),
        "total_models": len(predictions),
        "predictions": predictions,
        "skill_gap": skill_gap,
        "experience_gap": exp_gap,
        "years_experience": float(years_experience),
        "years_required": float(years_required),
        "tfidf_similarity": round(tfidf_sim, 4),
        "category_match": candidate_category.upper() == job_category.upper(),
        "candidate_category": candidate_category,
        "job_category": job_category
    }

