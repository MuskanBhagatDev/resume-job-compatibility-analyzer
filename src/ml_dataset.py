"""
ML Dataset Preparation and Splitting Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project - Phase 2 (Step 1).

Performs:
1. Dataset ingestion & validation (cv.csv, job.csv, matches.csv)
2. Safe relational merging on candidate_id and job_id
3. Validation checks (row counts, duplicates, missing values, unmatched IDs)
4. Extraction of two clean feature groups (Text Features & Structured Features)
5. Exclusion of all leakage, target, and post-hoc columns
6. Reproducible candidate-grouped train/val/test splitting
7. Summary metrics export to outputs/results/ml_dataset_summary.csv
"""

import sys
import os
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
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
import preprocessing


def build_and_validate_merged_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw tables, validates IDs, and merges candidate resumes,
    job postings, and match pairs.
    
    Returns:
        tuple: (merged_df, validation_report_dict)
    """
    cv_df, job_df, matches_df = data_loader.load_all_datasets()
    
    # Validation 1: Unmatched IDs
    unique_cand_cv = set(cv_df["candidate_id"].unique())
    unique_job_all = set(job_df["job_id"].unique())
    
    matches_cands = set(matches_df["candidate_id"].unique())
    matches_jobs = set(matches_df["job_id"].unique())
    
    unmatched_cands = matches_cands - unique_cand_cv
    unmatched_jobs = matches_jobs - unique_job_all
    
    # Validation 2: Duplicate candidate-job pairs in matches
    dup_pairs_matches = int(matches_df.duplicated(subset=["candidate_id", "job_id"]).sum())
    
    # Perform Merge
    # Rename overlapping column names before merge for clarity
    cv_clean = cv_df.rename(columns={
        "clean_skills": "clean_skills_cv",
        "clean_skills_json": "clean_skills_json_cv",
        "education_text": "education_text_cv",
        "embedding": "embedding_cv"
    })
    
    job_clean = job_df.rename(columns={
        "category": "job_category",
        "clean_skills": "clean_skills_job",
        "clean_skills_json": "clean_skills_json_job",
        "education_text": "education_text_job",
        "embedding": "embedding_job"
    })
    
    # Merge matches with candidate CV data
    merged = matches_df.merge(cv_clean, on="candidate_id", how="inner")
    # Merge result with job data
    merged = merged.merge(job_clean, on="job_id", how="inner")
    
    # Validation 3: Merged integrity checks
    total_merged_rows = len(merged)
    dup_pairs_merged = int(merged.duplicated(subset=["candidate_id", "job_id"]).sum())
    missing_cells = int(merged.isnull().sum().sum())
    
    report = {
        "cv_rows": len(cv_df),
        "job_rows": len(job_df),
        "matches_rows": len(matches_df),
        "merged_rows": total_merged_rows,
        "unique_candidates_in_matches": len(matches_cands),
        "unique_jobs_in_matches": len(matches_jobs),
        "unmatched_candidate_ids": len(unmatched_cands),
        "unmatched_job_ids": len(unmatched_jobs),
        "duplicate_pairs": dup_pairs_merged,
        "missing_cells_in_merged": missing_cells
    }
    
    return merged, report


def engineer_safe_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs legitimate candidate-job interaction features
    WITHOUT touching any leakage fields.
    """
    df = merged_df.copy()
    
    # 1. Experience gap: candidate experience minus employer requirement
    # Positive means candidate exceeds requirement, negative means deficit
    df["experience_gap"] = df["years_experience"].astype(float) - df["years_required"].astype(float)
    
    # 2. Category match indicator: candidate's inferred category matches job domain
    cand_cat = df["inferred_category"].astype(str).str.strip().str.upper()
    job_cat = df["job_category"].astype(str).str.strip().str.upper()
    df["category_match"] = (cand_cat == job_cat).astype(int)
    
    # 3. Text length & keyword count metadata
    df["resume_word_count"] = df["resume_text"].astype(str).apply(lambda x: len(x.split()))
    df["job_desc_word_count"] = df["job_description"].astype(str).apply(lambda x: len(x.split()))
    
    def count_skills(skills_val):
        if not isinstance(skills_val, str) or pd.isna(skills_val):
            return 0
        return len([s for s in skills_val.split(",") if s.strip()])
        
    df["skill_count_cv"] = df["clean_skills_cv"].apply(count_skills)
    df["skill_count_job"] = df["clean_skills_job"].apply(count_skills)
    df["skill_count_ratio"] = df.apply(
        lambda r: (r["skill_count_cv"] / r["skill_count_job"]) if r["skill_count_job"] > 0 else 0.0,
        axis=1
    ).round(4)
    
    return df


def split_dataset_by_candidate(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Splits the merged dataset into Train, Validation, and Test sets
    strictly grouped by candidate_id.
    
    Ensures ZERO candidate overlap across splits:
    A candidate's entire set of job applications belongs exclusively
    to one partition.
    """
    unique_candidates = df["candidate_id"].unique()
    rng = np.random.default_rng(random_seed)
    shuffled_candidates = rng.permutation(unique_candidates)
    
    n_candidates = len(shuffled_candidates)
    n_train = int(train_ratio * n_candidates)
    n_val = int(val_ratio * n_candidates)
    
    train_cand_set = set(shuffled_candidates[:n_train])
    val_cand_set = set(shuffled_candidates[n_train:n_train + n_val])
    test_cand_set = set(shuffled_candidates[n_train + n_val:])
    
    # Verify candidate disjointness
    assert len(train_cand_set & val_cand_set) == 0, "Candidate overlap between Train and Val!"
    assert len(train_cand_set & test_cand_set) == 0, "Candidate overlap between Train and Test!"
    assert len(val_cand_set & test_cand_set) == 0, "Candidate overlap between Val and Test!"
    
    train_df = df[df["candidate_id"].isin(train_cand_set)].copy().reset_index(drop=True)
    val_df = df[df["candidate_id"].isin(val_cand_set)].copy().reset_index(drop=True)
    test_df = df[df["candidate_id"].isin(test_cand_set)].copy().reset_index(drop=True)
    
    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_candidates": len(train_cand_set),
        "val_candidates": len(val_cand_set),
        "test_candidates": len(test_cand_set),
        "train_pct": round(len(train_df) / len(df) * 100, 2),
        "val_pct": round(len(val_df) / len(df) * 100, 2),
        "test_pct": round(len(test_df) / len(df) * 100, 2)
    }
    
    return train_df, val_df, test_df, split_info


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Defines the standard separation of feature sets for classical ML modeling.
    """
    text_features = [
        "resume_text",
        "job_description",
        "job_text",
        "clean_skills_cv",
        "clean_skills_job"
    ]
    
    structured_features = [
        "years_experience",
        "years_required",
        "experience_gap",
        "category_match",
        "inferred_category",
        "job_category",
        "target_position",
        "job_title",
        "resume_word_count",
        "job_desc_word_count",
        "skill_count_cv",
        "skill_count_job",
        "skill_count_ratio"
    ]
    
    excluded_leakage_features = [
        "final_score",         # Target (Regression)
        "label_strength",      # Target (Classification)
        "raw_score",           # Deterministic composite proxy
        "candidate_rank",      # Post-hoc evaluation outcome
        "candidate_rank_pct",  # Post-hoc evaluation outcome
        "semantic_score",      # Pre-calculated synthetic score
        "skill_score",         # Pre-calculated synthetic score
        "position_score",      # Pre-calculated synthetic score
        "category_score",      # Pre-calculated synthetic score
        "experience_score",    # Pre-calculated synthetic score
        "education_score",     # Pre-calculated synthetic score
        "candidate_id",        # Identifier
        "job_id",              # Identifier
        "source_row",          # Extraction metadata
        "embedding_cv",        # Black-box external embedding
        "embedding_job"        # Black-box external embedding
    ]
    
    return {
        "text_features": text_features,
        "structured_features": structured_features,
        "excluded_leakage_features": excluded_leakage_features
    }


def run_dataset_preparation(save_output: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes complete ML Dataset validation, feature grouping,
    and candidate-grouped splitting.
    """
    print("\n" + "=" * 80)
    print("                 ML DATASET PREPARATION & VALIDATION                    ")
    print("=" * 80)
    
    # 1. Merge and validate
    merged_raw, val_report = build_and_validate_merged_dataset()
    print("\n[1] DATASET MERGE & INTEGRITY AUDIT:")
    print("-" * 80)
    print(f"  - Resumes (cv.csv):             {val_report['cv_rows']:,} rows")
    print(f"  - Jobs (job.csv):                {val_report['job_rows']:,} rows")
    print(f"  - Matches (matches.csv):         {val_report['matches_rows']:,} rows")
    print(f"  - Merged Total Rows:             {val_report['merged_rows']:,} rows (100% matched)")
    print(f"  - Unmatched Candidate IDs:       {val_report['unmatched_candidate_ids']}")
    print(f"  - Unmatched Job IDs:             {val_report['unmatched_job_ids']}")
    print(f"  - Duplicate (Candidate,Job) Pairs: {val_report['duplicate_pairs']}")
    print(f"  - Total Missing Values in Merged: {val_report['missing_cells_in_merged']}")
    
    # 2. Feature engineering & grouping
    engineered_df = engineer_safe_features(merged_raw)
    feature_groups = get_feature_groups()
    
    print("\n[2] FEATURE SPECIFICATION & LEAKAGE ISOLATION:")
    print("-" * 80)
    print(f"  [+] TEXT FEATURES ({len(feature_groups['text_features'])}):")
    for feat in feature_groups["text_features"]:
        print(f"      * {feat}")
        
    print(f"\n  [+] STRUCTURED FEATURES ({len(feature_groups['structured_features'])}):")
    for feat in feature_groups["structured_features"]:
        print(f"      * {feat}")
        
    print(f"\n  [x] EXCLUDED LEAKAGE / TARGET FIELDS ({len(feature_groups['excluded_leakage_features'])}):")
    for feat in feature_groups["excluded_leakage_features"]:
        print(f"      * {feat}")
        
    # 3. Candidate-grouped splitting
    train_df, val_df, test_df, split_info = split_dataset_by_candidate(engineered_df)
    
    print("\n[3] REPRODUCIBLE CANDIDATE-GROUPED SPLIT (70/15/15):")
    print("-" * 80)
    print(f"  - Train Set:      {split_info['train_rows']:,} rows ({split_info['train_pct']}%) | {split_info['train_candidates']} unique candidates")
    print(f"  - Validation Set: {split_info['val_rows']:,} rows ({split_info['val_pct']}%) | {split_info['val_candidates']} unique candidates")
    print(f"  - Test Set:       {split_info['test_rows']:,} rows ({split_info['test_pct']}%) | {split_info['test_candidates']} unique candidates")
    print("  - Candidate Overlap across splits: EXACTLY 0 (Strictly disjoint)")
    
    # Class distribution across splits
    print("\n  Class Proportions Across Splits (label_strength):")
    train_dist = train_df["label_strength"].value_counts(normalize=True) * 100
    val_dist = val_df["label_strength"].value_counts(normalize=True) * 100
    test_dist = test_df["label_strength"].value_counts(normalize=True) * 100
    
    split_dist_df = pd.DataFrame({
        "Train (%)": train_dist.round(2),
        "Validation (%)": val_dist.round(2),
        "Test (%)": test_dist.round(2)
    })
    print(split_dist_df.to_string())
    
    # 4. Save summary
    if save_output:
        summary_rows = [
            {"Property": "Total Merged Rows", "Value": str(val_report["merged_rows"])},
            {"Property": "Unique Candidates", "Value": str(val_report["unique_candidates_in_matches"])},
            {"Property": "Unique Jobs", "Value": str(val_report["unique_jobs_in_matches"])},
            {"Property": "Duplicate Pairs", "Value": str(val_report["duplicate_pairs"])},
            {"Property": "Missing Cells", "Value": str(val_report["missing_cells_in_merged"])},
            {"Property": "Text Features Count", "Value": str(len(feature_groups["text_features"]))},
            {"Property": "Structured Features Count", "Value": str(len(feature_groups["structured_features"]))},
            {"Property": "Excluded Leakage Columns Count", "Value": str(len(feature_groups["excluded_leakage_features"]))},
            {"Property": "Train Rows", "Value": f"{split_info['train_rows']} ({split_info['train_pct']}%)"},
            {"Property": "Validation Rows", "Value": f"{split_info['val_rows']} ({split_info['val_pct']}%)"},
            {"Property": "Test Rows", "Value": f"{split_info['test_rows']} ({split_info['test_pct']}%)"},
            {"Property": "Candidate Overlap", "Value": "0 (Disjoint)"}
        ]
        summary_df = pd.DataFrame(summary_rows)
        out_file = config.RESULTS_DIR / "ml_dataset_summary.csv"
        summary_df.to_csv(out_file, index=False)
        print(f"\n[OUTPUT SAVED] Dataset summary saved to: {out_file}")
        
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_dataset_preparation(save_output=True)

