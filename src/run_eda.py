"""
Master Execution Script for Phase 1: Dataset Loading and EDA Pipeline.
Resume Job Compatibility Analyzer
Academic ML Lab Mini-Project

Executes:
1. Loading of all three CSV files (cv.csv, job.csv, matches.csv)
2. Printing shapes and schema details
3. Displaying preview of first 5 rows
4. Checking missing values (counts and percentages)
5. Checking duplicate records
6. Displaying descriptive summary statistics
7. Analyzing Job category distribution
8. Analyzing Candidate category distribution
9. Analyzing Match compatibility label distribution
10. Generating informative EDA plots with Matplotlib
11. Analyzing final_score distribution metrics
12. Analyzing component scores (semantic, skill, experience, education, category)
13. Saving all plots inside outputs/plots/
"""

import sys
import os
from pathlib import Path

# Add src to Python path for seamless importing
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import data_loader
import eda_plots


def main():
    print("=" * 80)
    print("      RESUME JOB COMPATIBILITY ANALYZER - PHASE 1: DATA LOAD & EDA      ")
    print("                      Academic ML Lab Mini-Project                      ")
    print("=" * 80)

    # 1. Load all three CSV files
    print("\n[STEP 1] Loading CSV datasets from data/ directory...")
    cv_df, job_df, matches_df = data_loader.load_all_datasets()
    
    # 2. Print shapes
    print("\n[STEP 2] Dataset Dimensions (Shapes):")
    print(f"  - Resumes (cv.csv):    {cv_df.shape[0]:,} rows x {cv_df.shape[1]} columns")
    print(f"  - Jobs (job.csv):       {job_df.shape[0]:,} rows x {job_df.shape[1]} columns")
    print(f"  - Matches (matches.csv): {matches_df.shape[0]:,} rows x {matches_df.shape[1]} columns")

    # 3, 4, 5, 6. Inspect datasets (head, missing values, duplicates, statistics)
    print("\n[STEP 3 to 6] Detailed Data Inspection & Statistics:")
    data_loader.inspect_dataset(cv_df, "Candidate CVs (cv.csv)")
    data_loader.inspect_dataset(job_df, "Job Postings (job.csv)")
    data_loader.inspect_dataset(matches_df, "Match Pairs (matches.csv)")

    # 7. Job category distribution
    print("\n" + "=" * 80)
    print("[STEP 7] Job Category Distribution:")
    print("=" * 80)
    job_cat = data_loader.get_job_category_distribution(job_df)
    print(job_cat.to_string(index=False))

    # 8. Candidate category distribution
    print("\n" + "=" * 80)
    print("[STEP 8] Candidate Inferred Category Distribution:")
    print("=" * 80)
    cv_cat = data_loader.get_candidate_category_distribution(cv_df)
    print(cv_cat.to_string(index=False))

    # 9. Compatibility label distribution
    print("\n" + "=" * 80)
    print("[STEP 9] Compatibility Label Strength Distribution:")
    print("=" * 80)
    label_dist = data_loader.get_label_distribution(matches_df)
    print(label_dist.to_string(index=False))

    # 11 & 12. Analyze final_score and individual component scores
    print("\n" + "=" * 80)
    print("[STEP 11 & 12] Compatibility Score Statistics & Distribution Analysis:")
    print("=" * 80)
    score_stats = data_loader.get_score_analysis(matches_df)
    print(score_stats.to_string())

    # Save summary stats table to outputs/results/
    summary_csv_path = config.RESULTS_DIR / "eda_score_summary.csv"
    score_stats.to_csv(summary_csv_path)
    print(f"\n[INFO] Saved descriptive statistics to: {summary_csv_path}")

    # 10 & 13. Plot useful EDA graphs and save inside outputs/plots/
    print("\n" + "=" * 80)
    print("[STEP 10 & 13] Generating and Saving EDA Visualizations (outputs/plots/)...")
    print("=" * 80)
    plots = eda_plots.generate_all_plots(cv_df, job_df, matches_df)

    print("\n" + "=" * 80)
    print("                      PHASE 1 EXECUTION COMPLETE!                      ")
    print("=" * 80)
    print(f"Generated {len(plots)} plots saved in: {config.PLOTS_DIR}")
    for p in plots:
        print(f"  [+] {p.name}")
    print("\nAll dataset checks, descriptive statistics, and EDA figures are ready for academic review.")



if __name__ == "__main__":
    main()
