"""
EDA Plotting Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project.

Generates and saves exploratory data analysis plots using Matplotlib:
1. Job category distribution
2. Candidate category distribution
3. Job vs. Candidate category proportion comparison
4. Match compatibility label distribution (Bar + Donut)
5. Final score distribution (Histogram + Boxplot)
6. Component scores distributions (Semantic, Skill, Experience, Education, Category)
7. Score correlation heatmap
8. Component scores stratified by match label strength
9. Experience distribution (Candidate vs. Job Required)
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import configurations
try:
    from . import config
except ImportError:
    import config


def set_plot_style():
    """Configure matplotlib defaults for clean, professional academic figures."""
    plt.style.use(config.PLOT_STYLE)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["figure.titleweight"] = "bold"


def plot_job_category_distribution(job_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot distribution of job categories."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "01_job_category_distribution.png")
    
    col = "category" if "category" in job_df.columns else job_df.columns[1]
    counts = job_df[col].value_counts()
    percentages = (counts / counts.sum()) * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(counts.index, counts.values, color="#2b5c8f", edgecolor="black", alpha=0.85)
    
    # Annotate bars with counts and percentages
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        width = bar.get_width()
        ax.text(width + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", ha="left", fontsize=9, fontweight="bold")
        
    ax.set_title("Job Postings Distribution by Category", pad=15)
    ax.set_xlabel("Number of Job Postings")
    ax.set_ylabel("Job Category")
    ax.set_xlim(0, max(counts.values) * 1.22)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_candidate_category_distribution(cv_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot distribution of candidate categories."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "02_candidate_category_distribution.png")
    
    col = "inferred_category" if "inferred_category" in cv_df.columns else cv_df.columns[-1]
    counts = cv_df[col].value_counts()
    percentages = (counts / counts.sum()) * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(counts.index, counts.values, color="#1e7b5e", edgecolor="black", alpha=0.85)
    
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        width = bar.get_width()
        ax.text(width + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", ha="left", fontsize=9, fontweight="bold")
        
    ax.set_title("Candidate Profiles Distribution by Inferred Category", pad=15)
    ax.set_xlabel("Number of Candidates")
    ax.set_ylabel("Inferred Domain Category")
    ax.set_xlim(0, max(counts.values) * 1.25)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_category_comparison(job_df: pd.DataFrame, cv_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot side-by-side percentage comparison of Job vs. Candidate categories."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "03_category_comparison.png")
    
    job_col = "category" if "category" in job_df.columns else job_df.columns[1]
    cv_col = "inferred_category" if "inferred_category" in cv_df.columns else cv_df.columns[-1]
    
    job_pct = (job_df[job_col].value_counts(normalize=True) * 100)
    cv_pct = (cv_df[cv_col].value_counts(normalize=True) * 100)
    
    categories = sorted(list(set(job_pct.index).union(set(cv_pct.index))))
    job_vals = [job_pct.get(c, 0.0) for c in categories]
    cv_vals = [cv_pct.get(c, 0.0) for c in categories]
    
    y = np.arange(len(categories))
    bar_height = 0.38
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.barh(y - bar_height/2, job_vals, height=bar_height, label="Job Openings (%)", color="#2b5c8f", edgecolor="black", alpha=0.85)
    bars2 = ax.barh(y + bar_height/2, cv_vals, height=bar_height, label="Candidates (%)", color="#e07a5f", edgecolor="black", alpha=0.85)
    
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Proportion of Dataset (%)")
    ax.set_title("Category Distribution Comparison: Jobs vs. Candidates", pad=15)
    ax.legend(loc="lower right", frameon=True)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_compatibility_label_distribution(matches_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot compatibility label strength distribution (Bar and Donut charts)."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "04_compatibility_label_distribution.png")
    
    col = "label_strength" if "label_strength" in matches_df.columns else "label"
    counts = matches_df[col].value_counts()
    percentages = (counts / counts.sum()) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2b5c8f", "#e07a5f", "#3d8b4f"]
    if len(counts) > 3:
        colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))
    
    # 1. Bar Chart
    bars = ax1.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor="black", alpha=0.85)
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + max(counts.values) * 0.015,
                 f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_title("Frequency of Match Compatibility Classes", pad=12)
    ax1.set_ylabel("Match Pair Count")
    ax1.set_xlabel("Compatibility Strength Label")
    ax1.set_ylim(0, max(counts.values) * 1.2)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # 2. Donut Chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors[:len(counts)],
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        pctdistance=0.75
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(9)
        autotext.set_weight("bold")
    ax2.set_title("Proportion of Match Categories", pad=12)
    
    fig.suptitle("Match Pair Label Strength Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_final_score_distribution(matches_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot distribution of overall match final_score (Histogram + Boxplot)."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "05_final_score_distribution.png")
    
    scores = matches_df["final_score"].dropna()
    mean_val = scores.mean()
    median_val = scores.median()
    std_val = scores.std()
    
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [0.25, 0.75], "hspace": 0.08}
    )
    
    # Top: Boxplot
    ax_box.boxplot(scores, vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#7aa6c2", edgecolor="#1a4066", alpha=0.8),
                   medianprops=dict(color="#d9381e", linewidth=2),
                   whiskerprops=dict(color="#1a4066", linewidth=1.5),
                   capprops=dict(color="#1a4066", linewidth=1.5),
                   flierprops=dict(marker="o", markersize=3, alpha=0.3, markerfacecolor="gray"))
    ax_box.set_yticks([])
    ax_box.grid(axis="x", linestyle="--", alpha=0.5)
    ax_box.set_title("Overall Compatibility Final Score Distribution", pad=10)
    
    # Bottom: Histogram
    counts, bins, _ = ax_hist.hist(scores, bins=50, color="#2b5c8f", edgecolor="black", alpha=0.75, density=True)
    
    # Statistical lines
    ax_hist.axvline(mean_val, color="#d9381e", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.3f}")
    ax_hist.axvline(median_val, color="#2ca02c", linestyle="-", linewidth=2, label=f"Median: {median_val:.3f}")
    ax_hist.axvline(mean_val - std_val, color="orange", linestyle=":", linewidth=1.5, label=f"±1 Std ({std_val:.3f})")
    ax_hist.axvline(mean_val + std_val, color="orange", linestyle=":", linewidth=1.5)
    
    ax_hist.set_xlabel("Final Match Score [0.0 - 1.0]")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.legend(loc="upper right", frameon=True)
    ax_hist.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_component_scores_distribution(matches_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot multi-panel distributions of all individual match component scores."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "06_component_scores_distributions.png")
    
    components = [
        ("semantic_score", "Semantic Compatibility", "#1f77b4"),
        ("skill_score", "Skill Set Overlap", "#2ca02c"),
        ("experience_score", "Experience Alignment", "#ff7f0e"),
        ("education_score", "Education Qualification", "#9467bd"),
        ("category_score", "Domain/Category Match", "#d62728")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    for idx, (col, title, color) in enumerate(components):
        ax = axes[idx]
        if col in matches_df.columns:
            data = matches_df[col].dropna()
            ax.hist(data, bins=35, color=color, edgecolor="black", alpha=0.75)
            mean_c = data.mean()
            median_c = data.median()
            ax.axvline(mean_c, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_c:.2f}")
            ax.axvline(median_c, color="red", linestyle=":", linewidth=1.5, label=f"Med: {median_c:.2f}")
            ax.set_title(title, pad=8)
            ax.set_xlabel("Score")
            ax.set_ylabel("Pair Count")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
            
    # Sixth subplot: Comparative boxplot of all 5 components
    ax_all = axes[5]
    valid_cols = [c[0] for c in components if c[0] in matches_df.columns]
    short_labels = ["Semantic", "Skill", "Experience", "Education", "Category"]
    box_data = [matches_df[c].dropna() for c in valid_cols]
    bp = ax_all.boxplot(box_data, patch_artist=True, tick_labels=short_labels)
    
    for patch, (_, _, color) in zip(bp["boxes"], components):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        
    ax_all.set_title("Comparative Component Summary", pad=8)
    ax_all.set_ylabel("Score Range")
    ax_all.tick_params(axis="x", rotation=25)
    ax_all.grid(axis="y", linestyle="--", alpha=0.4)
    
    fig.suptitle("Resume-Job Feature Scores Breakdown", fontsize=15, y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_score_correlation_heatmap(matches_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot correlation heatmap among final_score, raw_score, and subcomponents."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "07_score_correlation_heatmap.png")
    
    cols = [
        "final_score", "raw_score", "semantic_score",
        "skill_score", "position_score", "category_score",
        "experience_score", "education_score"
    ]
    valid_cols = [c for c in cols if c in matches_df.columns]
    corr_matrix = matches_df[valid_cols].corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(cax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson Correlation Coefficient", rotation=-90, va="bottom")
    
    # Set tick labels
    ax.set_xticks(range(len(valid_cols)))
    ax.set_yticks(range(len(valid_cols)))
    clean_labels = [c.replace("_", " ").title() for c in valid_cols]
    ax.set_xticklabels(clean_labels, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(clean_labels, fontsize=9)
    
    # Annotate matrix values
    for i in range(len(valid_cols)):
        for j in range(len(valid_cols)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)
            
    ax.set_title("Correlation Heatmap: Match Metrics & Sub-Scores", pad=35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_scores_by_label_strength(matches_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot distribution of key scores stratified by match label strength."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "08_scores_by_label_strength.png")
    
    labels = matches_df["label_strength"].unique().tolist()
    labels = sorted(labels)
    
    metrics = ["final_score", "semantic_score", "skill_score", "experience_score"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data_by_label = [matches_df[matches_df["label_strength"] == lbl][metric].dropna() for lbl in labels]
        bp = ax.boxplot(data_by_label, patch_artist=True, tick_labels=[l.replace("_", " ").title() for l in labels])
        
        box_colors = ["#e07a5f", "#3d8b4f", "#2b5c8f"]
        for patch, c in zip(bp["boxes"], box_colors[:len(labels)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
            
        ax.set_title(metric.replace("_", " ").title(), pad=10)
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
    fig.suptitle("Score Distributions Stratified by Compatibility Label", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_experience_distribution(cv_df: pd.DataFrame, job_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot distribution of candidate years_experience vs job years_required."""
    set_plot_style()
    output_path = save_path or (config.PLOTS_DIR / "09_experience_distribution.png")
    
    cand_exp = cv_df["years_experience"].dropna() if "years_experience" in cv_df.columns else pd.Series()
    job_exp = job_df["years_required"].dropna() if "years_required" in job_df.columns else pd.Series()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Candidates
    if not cand_exp.empty:
        ax1.hist(cand_exp, bins=25, color="#1e7b5e", edgecolor="black", alpha=0.8)
        ax1.axvline(cand_exp.median(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Median: {cand_exp.median():.1f} yrs")
        ax1.set_title("Candidate Experience Distribution", pad=12)
        ax1.set_xlabel("Years of Experience")
        ax1.set_ylabel("Candidate Count")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.4)
        
    # Jobs
    if not job_exp.empty:
        ax2.hist(job_exp, bins=20, color="#2b5c8f", edgecolor="black", alpha=0.8)
        ax2.axvline(job_exp.median(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Median: {job_exp.median():.1f} yrs")
        ax2.set_title("Job Openings Experience Requirement", pad=12)
        ax2.set_xlabel("Years Required")
        ax2.set_ylabel("Job Count")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.4)
        
    fig.suptitle("Experience Profile: Talent Supply vs. Employer Demand", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def generate_all_plots(cv_df: pd.DataFrame, job_df: pd.DataFrame, matches_df: pd.DataFrame):
    """Generate and save the complete suite of EDA visualizations."""
    print("\n" + "="*70)
    print(" GENERATING EXPLORATORY DATA ANALYSIS (EDA) PLOTS")
    print("="*70)
    
    p1 = plot_job_category_distribution(job_df)
    p2 = plot_candidate_category_distribution(cv_df)
    p3 = plot_category_comparison(job_df, cv_df)
    p4 = plot_compatibility_label_distribution(matches_df)
    p5 = plot_final_score_distribution(matches_df)
    p6 = plot_component_scores_distribution(matches_df)
    p7 = plot_score_correlation_heatmap(matches_df)
    p8 = plot_scores_by_label_strength(matches_df)
    p9 = plot_experience_distribution(cv_df, job_df)
    
    print(f"\n[COMPLETE] Successfully generated and saved 9 EDA plots in: {config.PLOTS_DIR}")
    return [p1, p2, p3, p4, p5, p6, p7, p8, p9]

