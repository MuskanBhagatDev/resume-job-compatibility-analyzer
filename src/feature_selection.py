"""
Feature Selection Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project - Phase 2 (Step 2 - Experiment 1).

Implements:
1. SelectKBest feature selection using ANOVA F-score (f_classif)
2. Train-only fitting to strictly prevent data leakage
3. Top-k feature ranking, score attribution, and visualization
4. Exports feature selection metrics to outputs/results/feature_selection_summary.csv
5. Generates feature importance bar chart in outputs/plots/feature_selection_importance.png
6. Persists fitted selector to models/feature_selector.joblib
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif

# Ensure path resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def fit_feature_selector(
    X_train: Any,
    y_train: np.ndarray,
    feature_names: List[str],
    k: int = 500
) -> Tuple[SelectKBest, pd.DataFrame]:
    """
    Fits SelectKBest(f_classif) exclusively on the training partition.
    
    Parameters:
        X_train: Sparse or dense feature matrix (n_samples, n_features)
        y_train: Classification labels (n_samples,)
        feature_names: List of all input feature names
        k: Number of top features to retain
        
    Returns:
        tuple: (fitted_selector, ranking_dataframe)
    """
    total_features = X_train.shape[1]
    actual_k = min(k, total_features)
    print(f"[FEATURE SELECTION] Running SelectKBest (f_classif, k={actual_k} of {total_features})...")
    
    selector = SelectKBest(score_func=f_classif, k=actual_k)
    selector.fit(X_train, y_train)
    
    scores = selector.scores_
    pvalues = selector.pvalues_
    
    # Handle NaN or Inf in scores safely
    clean_scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    clean_pvals = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=1.0)
    
    ranking_df = pd.DataFrame({
        "Feature_Index": np.arange(total_features),
        "Feature_Name": feature_names if len(feature_names) == total_features else [f"feature_{i}" for i in range(total_features)],
        "F_Score": clean_scores.round(4),
        "P_Value": clean_pvals,
        "Selected": selector.get_support()
    }).sort_values(by="F_Score", ascending=False).reset_index(drop=True)
    
    ranking_df["Rank"] = np.arange(1, len(ranking_df) + 1)
    return selector, ranking_df


def plot_top_features(ranking_df: pd.DataFrame, top_n: int = 20, save_path: Optional[Path] = None) -> Path:
    """Plots top-n most discriminative features selected by ANOVA F-score."""
    output_path = save_path or (config.PLOTS_DIR / "feature_selection_importance.png")
    top_df = ranking_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar chart
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top_df["Feature_Name"], top_df["F_Score"], color="#2b5c8f", edgecolor="black", alpha=0.85)
    
    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(top_df["F_Score"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{width:,.1f}", va="center", ha="left", fontsize=9, fontweight="bold")
                
    ax.set_title(f"Top {top_n} Discriminative Features Selected by ANOVA F-Score", pad=15, fontsize=12, fontweight="bold")
    ax.set_xlabel("ANOVA F-Statistic Score (Correlation with Target Compatibility)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Engineered Feature / Token", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(top_df["F_Score"]) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def run_feature_selection_experiment(
    X_train: Any,
    y_train: np.ndarray,
    feature_names: List[str],
    k: int = 500,
    save_outputs: bool = True
) -> Tuple[SelectKBest, pd.DataFrame]:
    """
    Executes full feature selection experiment and persists outputs.
    """
    selector, ranking_df = fit_feature_selector(X_train, y_train, feature_names, k=k)
    
    print("\n" + "=" * 80)
    print("             FEATURE SELECTION (EXPERIMENT 1 - SelectKBest)             ")
    print("=" * 80)
    print(f"Total Candidate Features: {len(ranking_df):,}")
    print(f"Features Retained (k):     {k:,}")
    print("\nTop 15 Most Discriminative Features:")
    print("-" * 80)
    print(ranking_df[["Rank", "Feature_Name", "F_Score", "P_Value", "Selected"]].head(15).to_string(index=False))
    
    if save_outputs:
        summary_csv = config.RESULTS_DIR / "feature_selection_summary.csv"
        ranking_df.to_csv(summary_csv, index=False)
        print(f"\n[OUTPUT SAVED] Full ranking saved to: {summary_csv}")
        
        plot_top_features(ranking_df, top_n=20)
        
        selector_path = config.MODELS_DIR / "feature_selector.joblib"
        joblib.dump(selector, selector_path)
        print(f"[MODEL SAVED] Selector object saved to: {selector_path}")
        
    return selector, ranking_df

