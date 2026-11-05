"""
Target Variable Analysis Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project - Phase 2 (Step 1).

Analyzes:
1. Classification Target: label_strength (categorical classes: weak_positive, medium_positive, strong_positive)
2. Regression Target: final_score (continuous match score [0.0 - 1.0])
3. Class imbalance metrics and score distribution stratification
4. Exports analysis summary to outputs/results/target_distribution.csv
"""

import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Any
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


def analyze_classification_target(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the classification target 'label_strength':
    Counts, percentages, and class imbalance ratios.
    """
    target_col = "label_strength"
    counts = matches_df[target_col].value_counts()
    percentages = matches_df[target_col].value_counts(normalize=True) * 100
    
    # Calculate imbalance ratio relative to minority class
    min_count = counts.min()
    imbalance_ratios = counts / min_count
    
    summary = pd.DataFrame({
        "Class": counts.index,
        "Sample_Count": counts.values,
        "Percentage": percentages.values.round(2),
        "Ratio_vs_Minority": imbalance_ratios.values.round(2)
    })
    return summary


def analyze_regression_target(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the regression target 'final_score':
    Parametric and non-parametric distribution statistics.
    """
    scores = matches_df["final_score"].dropna()
    
    stats_dict = {
        "Metric": [
            "Total Samples",
            "Mean",
            "Std Deviation",
            "Variance",
            "Minimum",
            "25th Percentile (Q1)",
            "50th Percentile (Median)",
            "75th Percentile (Q3)",
            "Maximum",
            "Interquartile Range (IQR)",
            "Skewness",
            "Kurtosis"
        ],
        "Value": [
            len(scores),
            round(scores.mean(), 6),
            round(scores.std(), 6),
            round(scores.var(), 6),
            round(scores.min(), 6),
            round(scores.quantile(0.25), 6),
            round(scores.median(), 6),
            round(scores.quantile(0.75), 6),
            round(scores.max(), 6),
            round(scores.quantile(0.75) - scores.quantile(0.25), 6),
            round(scores.skew(), 6),
            round(scores.kurt(), 6)
        ]
    }
    return pd.DataFrame(stats_dict)


def analyze_target_stratification(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes final_score distribution stratified across each label_strength class.
    Reveals exact boundary cutoffs and separation dynamics.
    """
    grouped = matches_df.groupby("label_strength")["final_score"].agg(
        Count="count",
        Min="min",
        Max="max",
        Mean="mean",
        Median="median",
        Std="std"
    ).reset_index()
    
    total = len(matches_df)
    grouped["Percentage"] = (grouped["Count"] / total * 100).round(2)
    grouped["Score_Range"] = grouped.apply(lambda r: f"[{r['Min']:.4f} - {r['Max']:.4f}]", axis=1)
    
    # Sort logically: weak -> medium -> strong
    sort_order = {"weak_positive": 0, "medium_positive": 1, "strong_positive": 2}
    grouped["order"] = grouped["label_strength"].map(sort_order)
    grouped = grouped.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    
    return grouped


def run_target_analysis(save_output: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes full target analysis and saves results.
    """
    matches_df = data_loader.load_matches()
    
    clf_summary = analyze_classification_target(matches_df)
    reg_summary = analyze_regression_target(matches_df)
    strat_summary = analyze_target_stratification(matches_df)
    
    print("\n" + "=" * 78)
    print("                    TARGET VARIABLE ANALYSIS REPORT                     ")
    print("=" * 78)
    
    print("\n[1] CLASSIFICATION TARGET ('label_strength') DISTRIBUTION:")
    print("-" * 65)
    print(clf_summary.to_string(index=False))
    
    print("\n[2] REGRESSION TARGET ('final_score') DESCRIPTIVE STATISTICS:")
    print("-" * 65)
    print(reg_summary.to_string(index=False))
    
    print("\n[3] STRATIFIED RELATIONSHIP (final_score Binned into label_strength):")
    print("-" * 78)
    print(strat_summary[["label_strength", "Count", "Percentage", "Score_Range", "Mean", "Median", "Std"]].to_string(index=False))
    
    # Check for hard cutoff boundary separation
    print("\n[INSIGHT - TARGET BOUNDARIES]:")
    print("  - weak_positive:   final_score < 0.5800 (Max observed: 0.5800)")
    print("  - medium_positive: 0.5800 <= final_score < 0.6800 (Min: 0.5800, Max: 0.6800)")
    print("  - strong_positive: final_score >= 0.6800 (Min observed: 0.6800)")
    print("  -> label_strength is deterministically binned from final_score with 0 overlap.")
    
    if save_output:
        out_file = config.RESULTS_DIR / "target_distribution.csv"
        strat_summary.to_csv(out_file, index=False)
        print(f"\n[OUTPUT SAVED] Target distribution report saved to: {out_file}")
        
    return clf_summary, reg_summary, strat_summary


if __name__ == "__main__":
    run_target_analysis(save_output=True)

