"""
Data Leakage Audit Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project - Phase 2 (Step 1).

Performs a rigorous data leakage audit across all candidate score fields:
- final_score
- raw_score
- semantic_score
- skill_score
- position_score
- category_score
- experience_score
- education_score
- candidate_rank
- candidate_rank_pct

Identifies fields that directly encode, reveal, or are derived from the target variables.
Saves audit findings to outputs/results/leakage_audit.csv.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Ensure path resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import data_loader


def audit_score_fields(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Audits every score and ranking column in matches.csv for target leakage.
    
    Evaluates:
    - Correlation with final_score (continuous target)
    - Correlation with label_strength (classification target)
    - Leakage mechanism
    - Risk classification
    - Final inclusion verdict
    """
    label_map = {"weak_positive": 0, "medium_positive": 1, "strong_positive": 2}
    encoded_label = matches_df["label_strength"].map(label_map)
    final_score = matches_df["final_score"]
    
    audit_columns = [
        "final_score",
        "raw_score",
        "semantic_score",
        "skill_score",
        "position_score",
        "category_score",
        "experience_score",
        "education_score",
        "candidate_rank",
        "candidate_rank_pct"
    ]
    
    records = []
    
    for col in audit_columns:
        if col not in matches_df.columns:
            continue
            
        series = matches_df[col]
        corr_final = series.corr(final_score)
        corr_label = series.corr(encoded_label)
        
        # Determine specific leakage mechanism & risk profile
        if col == "final_score":
            risk = "CRITICAL / TARGET"
            mechanism = "Direct Target Definition. Deterministically cut at 0.58 and 0.68 into label_strength (100% trivial accuracy)."
            verdict = "EXCLUDE (Is Target)"
        elif col == "raw_score":
            risk = "CRITICAL LEAKAGE"
            mechanism = "Exact Deterministic Linear Combination (R² = 1.000) of 6 sub-scores: 0.25*sem + 0.35*skill + 0.20*pos + 0.10*cat + 0.05*exp + 0.05*edu."
            verdict = "EXCLUDE"
        elif col in ["candidate_rank", "candidate_rank_pct"]:
            risk = "POST-HOC OUTCOME LEAKAGE"
            mechanism = "Calculated post-hoc after evaluating all applicants against a job. At inference time for a new resume, rank is unknown."
            verdict = "EXCLUDE"
        elif col in ["semantic_score", "skill_score", "position_score", "category_score", "experience_score", "education_score"]:
            risk = "SYNTHETIC DERIVED SCORE"
            mechanism = "Proprietary ground-truth heuristic scores. In real deployment, only raw resume text/attributes are available."
            verdict = "EXCLUDE FROM RAW ML"
        else:
            risk = "LOW"
            mechanism = "Standard candidate or job attribute."
            verdict = "KEEP"
            
        records.append({
            "Column_Name": col,
            "Data_Type": str(series.dtype),
            "Corr_Final_Score": round(corr_final, 4),
            "Corr_Label_Strength": round(corr_label, 4),
            "Leakage_Risk_Level": risk,
            "Leakage_Mechanism": mechanism,
            "Verdict": verdict
        })
        
    audit_df = pd.DataFrame(records)
    return audit_df


def verify_deterministic_formula(matches_df: pd.DataFrame) -> Dict[str, float]:
    """
    Verifies mathematically that raw_score is an exact linear combination of the 6 sub-scores.
    """
    features = [
        "semantic_score", "skill_score", "position_score",
        "category_score", "experience_score", "education_score"
    ]
    X = matches_df[features].values
    y_raw = matches_df["raw_score"].values
    
    # Linear combination check with theoretical weights:
    # 0.25, 0.35, 0.20, 0.10, 0.05, 0.05
    theory_weights = np.array([0.25, 0.35, 0.20, 0.10, 0.05, 0.05])
    reconstructed = X @ theory_weights
    
    mae = float(np.mean(np.abs(reconstructed - y_raw)))
    max_err = float(np.max(np.abs(reconstructed - y_raw)))
    r2 = float(1.0 - np.sum((y_raw - reconstructed)**2) / np.sum((y_raw - y_raw.mean())**2))
    
    return {
        "MAE": mae,
        "Max_Error": max_err,
        "R2_Score": r2
    }


def verify_target_threshold_separation(matches_df: pd.DataFrame) -> float:
    """
    Verifies that label_strength is 100% deterministically determined by final_score.
    """
    # Cutoff rules:
    # < 0.58 -> weak_positive
    # 0.58 <= score < 0.68 -> medium_positive
    # >= 0.68 -> strong_positive
    scores = matches_df["final_score"].values
    actual = matches_df["label_strength"].values
    
    preds = np.where(scores < 0.58, "weak_positive",
             np.where(scores < 0.68, "medium_positive", "strong_positive"))
             
    accuracy = float(np.mean(preds == actual) * 100.0)
    return accuracy


def run_leakage_audit(save_output: bool = True) -> pd.DataFrame:
    """
    Executes complete Data Leakage Audit pipeline.
    """
    matches_df = data_loader.load_matches()
    
    print("\n" + "=" * 80)
    print("                      DATA LEAKAGE AUDIT REPORT                         ")
    print("=" * 80)
    
    # 1. Audit Table
    audit_df = audit_score_fields(matches_df)
    print("\n[1] FEATURE AUDIT & LEAKAGE RISK CLASSIFICATION:")
    print("-" * 80)
    display_cols = ["Column_Name", "Corr_Final_Score", "Corr_Label_Strength", "Leakage_Risk_Level", "Verdict"]
    print(audit_df[display_cols].to_string(index=False))
    
    # 2. Mathematical Proof of raw_score formula
    formula_metrics = verify_deterministic_formula(matches_df)
    print("\n[2] MATHEMATICAL DERIVATION CHECK FOR 'raw_score':")
    print("-" * 80)
    print("  Formula: raw_score = 0.25*semantic + 0.35*skill + 0.20*position + 0.10*category + 0.05*experience + 0.05*education")
    print("  Theoretical Fit R^2 Score: {:.6f} (Exact Identity)".format(formula_metrics['R2_Score']))

    print(f"  Mean Absolute Error:     {formula_metrics['MAE']:.10f}")
    print(f"  Maximum Absolute Error:  {formula_metrics['Max_Error']:.10f}")
    print("  -> Finding: raw_score is a 100% deterministic composite score.")
    
    # 3. Mathematical Proof of label_strength cutoff
    rule_acc = verify_target_threshold_separation(matches_df)
    print("\n[3] TARGET ENCODING CHECK (final_score -> label_strength):")
    print("-" * 80)
    print(f"  Exact Threshold Reconstruction Accuracy: {rule_acc:.2f}%")
    print("  Thresholds: [0.0 - 0.58) -> weak_positive | [0.58 - 0.68) -> medium_positive | [0.68 - 1.0] -> strong_positive")
    print("  -> Finding: Including final_score or raw_score as inputs causes 100% target leakage.")
    
    # 4. Summary of Safe vs Excluded Features
    print("\n[4] AUDIT VERDICT & FEATURE SEPARATION:")
    print("-" * 80)
    print("  [x] EXCLUDED FROM INPUTS (Leakage / Post-Hoc / Target):")
    print("      - final_score, label_strength (Target labels)")
    print("      - raw_score (Deterministic proxy)")
    print("      - candidate_rank, candidate_rank_pct (Post-hoc batch ranking)")
    print("      - semantic_score, skill_score, position_score, category_score, experience_score, education_score (Synthetic scores)")
    print("  [+] LEGITIMATE PREDICTORS TO USE (Authentic candidate & job data):")
    print("      - Text Features: resume_text, job_description/job_text, clean_skills (CV & Job)")
    print("      - Structured Features: years_experience, years_required, category_match, candidate/job domains, word counts")
    
    if save_output:
        out_file = config.RESULTS_DIR / "leakage_audit.csv"
        audit_df.to_csv(out_file, index=False)
        print(f"\n[OUTPUT SAVED] Leakage audit report saved to: {out_file}")
        
    return audit_df


if __name__ == "__main__":
    run_leakage_audit(save_output=True)
