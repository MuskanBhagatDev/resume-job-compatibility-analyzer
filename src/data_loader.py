"""
Data Loader Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project.

Provides functions to:
1. Load dataset CSV files (cv.csv, job.csv, matches.csv)
2. Inspect shapes, schemas, and preview rows
3. Check missing values and duplicate records
4. Compute summary statistics
5. Analyze category and compatibility label distributions
6. Analyze final_score and individual component scores
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np

# Import configurations
try:
    from . import config
except ImportError:
    import config


def load_cv(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load candidate CV records."""
    path = filepath or config.CV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"CV dataset not found at: {path}")
    print(f"[DATA LOADER] Loading CVs from {path}...")
    df = pd.read_csv(path)
    return df


def load_jobs(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load job posting records."""
    path = filepath or config.JOB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Job dataset not found at: {path}")
    print(f"[DATA LOADER] Loading Jobs from {path}...")
    df = pd.read_csv(path)
    return df


def load_matches(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load resume-job match records."""
    path = filepath or config.MATCHES_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Matches dataset not found at: {path}")
    print(f"[DATA LOADER] Loading Matches from {path}...")
    df = pd.read_csv(path)
    return df


def load_all_datasets(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three datasets at once.
    
    Returns:
        tuple: (cv_df, job_df, matches_df)
    """
    base_dir = data_dir or config.DATA_DIR
    cv_path = base_dir / "cv.csv"
    job_path = base_dir / "job.csv"
    matches_path = base_dir / "matches.csv"
    
    cv_df = load_cv(cv_path)
    job_df = load_jobs(job_path)
    matches_df = load_matches(matches_path)
    
    return cv_df, job_df, matches_df


def inspect_dataset(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Performs basic inspection on a given DataFrame:
    - Shape (rows, columns)
    - Column list and data types
    - Missing values (counts and percentages)
    - Duplicate rows count
    - First 5 rows preview
    - Descriptive statistics for numerical fields
    
    Returns:
        dict: Inspection metrics summary.
    """
    print(f"\n{'='*70}")
    print(f" DATASET INSPECTION: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # 1. Shape
    rows, cols = df.shape
    print(f"[SHAPE] Rows: {rows:,} | Columns: {cols}")
    
    # 2. Duplicate records
    duplicates = int(df.duplicated().sum())
    dup_pct = (duplicates / rows) * 100 if rows > 0 else 0
    print(f"[DUPLICATES] {duplicates} duplicate rows ({dup_pct:.2f}%)")
    
    # 3. Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / rows) * 100
    missing_summary = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing %": missing_pct.round(2)
    })
    total_missing = int(missing_counts.sum())
    print(f"[MISSING VALUES] Total missing cells: {total_missing}")
    if total_missing > 0:
        print(missing_summary[missing_summary["Missing Count"] > 0])
    else:
        print("  -> Zero missing values detected across all columns.")
        
    # 4. Preview first 5 rows
    print(f"\n[FIRST 5 ROWS PREVIEW]:")
    print(df.head(5))
    
    # 5. Basic statistics for numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print(f"\n[NUMERICAL SUMMARY STATISTICS]:")
        print(df[num_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])
    
    return {
        "name": dataset_name,
        "rows": rows,
        "cols": cols,
        "duplicates": duplicates,
        "total_missing": total_missing,
        "columns": df.columns.tolist()
    }


def get_job_category_distribution(job_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frequency and percentage distribution of job categories.
    """
    col = "category" if "category" in job_df.columns else job_df.columns[1]
    counts = job_df[col].value_counts()
    percentages = (job_df[col].value_counts(normalize=True) * 100).round(2)
    summary = pd.DataFrame({
        "Category": counts.index,
        "Count": counts.values,
        "Percentage (%)": percentages.values
    }).reset_index(drop=True)
    return summary


def get_candidate_category_distribution(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frequency and percentage distribution of candidate inferred categories.
    """
    col = "inferred_category" if "inferred_category" in cv_df.columns else cv_df.columns[-1]
    counts = cv_df[col].value_counts()
    percentages = (cv_df[col].value_counts(normalize=True) * 100).round(2)
    summary = pd.DataFrame({
        "Inferred Category": counts.index,
        "Count": counts.values,
        "Percentage (%)": percentages.values
    }).reset_index(drop=True)
    return summary


def get_label_distribution(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frequency and percentage distribution of match label strength.
    """
    col = "label_strength" if "label_strength" in matches_df.columns else "label"
    counts = matches_df[col].value_counts()
    percentages = (matches_df[col].value_counts(normalize=True) * 100).round(2)
    summary = pd.DataFrame({
        "Label Strength": counts.index,
        "Count": counts.values,
        "Percentage (%)": percentages.values
    }).reset_index(drop=True)
    return summary


def get_score_analysis(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for final_score
    and individual component scores.
    """
    target_scores = [
        "final_score",
        "semantic_score",
        "skill_score",
        "experience_score",
        "education_score",
        "category_score"
    ]
    # Filter to existing columns
    valid_scores = [c for c in target_scores if c in matches_df.columns]
    
    stats_df = matches_df[valid_scores].describe().T
    stats_df["skew"] = matches_df[valid_scores].skew()
    stats_df["kurtosis"] = matches_df[valid_scores].kurt()
    
    return stats_df.round(4)


if __name__ == "__main__":
    # If run directly as a script, execute inspection workflow
    try:
        from .run_eda import main
    except (ImportError, ValueError):
        from run_eda import main
    main()

