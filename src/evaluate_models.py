"""
Master Model Evaluation and Benchmarking Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project - Phase 2 (Step 2 - Methodological Correction & Audit).

Orchestrates the entire Phase 2 Step 2 workflow with strict academic protocols:
1. Feature Engineering (TF-IDF, Jaccard, Cosine Similarity, Structured Attributes)
2. Feature Selection (SelectKBest ANOVA F-Score: k=500 features retained)
3. Multi-Model Classifier Training:
   - Experiment A: Classifiers using All Features (1,516 features)
   - Experiment B: Classifiers using SelectKBest Features (500 features)
   - Domain-specific Models: KNN & Random Forest on Structured Features, Naive Bayes on TF-IDF
4. Strict Validation Protocol:
   - Model selection performed strictly on Validation Data (Val_Macro_F1)
   - Test data evaluated for final unbiased generalization benchmark
5. Regression Training & Shrinkage Dynamics:
   - OLS vs Ridge across multiple alpha penalties (1.0, 100.0, 1000.0)
6. Comprehensive Visualizations & Export of Comparison CSVs
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure path resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import ml_dataset
import feature_engineering
import feature_selection
import train_classifiers
import train_regressor


def plot_confusion_matrices(
    results_df: pd.DataFrame,
    class_names: List[str],
    save_path: Optional[Path] = None
) -> Path:
    """
    Plots a multi-panel grid of confusion matrices for all evaluated classifiers.
    """
    output_path = save_path or (config.PLOTS_DIR / "confusion_matrices_classification.png")
    n_models = len(results_df)
    
    # 2 rows x 4 columns grid for up to 7-8 models
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    clean_labels = [c.replace("_", " ").title() for c in class_names]
    
    for idx, row in results_df.iterrows():
        ax = axes[idx]
        cm = row["Confusion_Matrix"]
        
        # Normalize by true class counts for display
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{row['Model']}\n(Val F1: {row['Val_Macro_F1']:.3f} | Test F1: {row['Test_Macro_F1']:.3f})", pad=8, fontsize=9, fontweight="bold")
        
        tick_marks = np.arange(len(clean_labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(clean_labels, rotation=25, ha="right", fontsize=8)
        ax.set_yticklabels(clean_labels, fontsize=8)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                raw_cnt = cm[i, j]
                pct = cm_norm[i, j] * 100
                text_col = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{raw_cnt:,}\n({pct:.1f}%)", ha="center", va="center", color=text_col, fontsize=7.5, fontweight="bold")
                
        ax.set_ylabel("True Class", fontsize=8, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=8, fontweight="bold")
        
    for k in range(n_models, len(axes)):
        fig.delaxes(axes[k])
        
    fig.suptitle("Test Set Confusion Matrices Across Classifier Configurations", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def plot_model_comparison_bars(
    results_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """
    Plots a multi-metric bar chart comparing Validation Macro F1 vs Test Macro F1.
    """
    output_path = save_path or (config.PLOTS_DIR / "model_comparison_classification.png")
    
    short_names = [
        m.replace("Logistic Regression", "LogReg")
         .replace("Support Vector Machine", "SVM")
         .replace("K-Nearest Neighbors", "KNN")
         .replace("Random Forest", "RF")
         .replace("Naive Bayes", "NB")
        for m in results_df["Model"]
    ]
    
    x = np.arange(len(short_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    b1 = ax.bar(x - width/2, results_df["Val_Macro_F1"], width, label="Validation Macro F1", color="#2b5c8f", edgecolor="black", alpha=0.85)
    b2 = ax.bar(x + width/2, results_df["Test_Macro_F1"], width, label="Test Macro F1", color="#e07a5f", edgecolor="black", alpha=0.85)
    
    for b in [b1, b2]:
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
                    
    ax.set_ylabel("Macro F1 Score [0.0 - 1.0]", fontsize=10, fontweight="bold")
    ax.set_title("Classifier Comparison: Validation Set (Model Selection) vs. Held-Out Test Set", pad=15, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8.5, fontweight="bold", rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def main():
    print("=" * 80)
    print("   RESUME JOB COMPATIBILITY ANALYZER — PHASE 2 STEP 2: AUDIT & EXECUTION")
    print("                      Academic ML Lab Mini-Project                      ")
    print("=" * 80)

    # 1. Dataset Partitioning (Grouped by Candidate ID)
    print("\n[STEP 1] Ingesting Merged Dataset & Candidate-Grouped Partitions (70/15/15)...")
    merged_raw, _ = ml_dataset.build_and_validate_merged_dataset()
    engineered_raw = ml_dataset.engineer_safe_features(merged_raw)
    train_df, val_df, test_df, split_info = ml_dataset.split_dataset_by_candidate(engineered_raw, random_seed=42)
    print(f"  -> Train: {len(train_df):,} rows | Validation: {len(val_df):,} rows | Test: {len(test_df):,} rows")

    # 2. Feature Engineering Pipeline
    print("\n[STEP 2] Fitting Feature Pipeline on Training Data Only (Leakage-Safe)...")
    t0 = time.time()
    pipeline = feature_engineering.CompatibilityFeaturePipeline(max_tfidf_features=1500)
    pipeline.fit(train_df)
    pipeline.save()
    print(f"  -> Pipeline fit completed in {time.time() - t0:.2f}s.")

    print("\n[STEP 3] Transforming Partitions (Train, Val, Test)...")
    train_feats = pipeline.transform(train_df)
    val_feats = pipeline.transform(val_df)
    test_feats = pipeline.transform(test_df)
    print("  -> Base Transformed Matrices:")
    print(f"     Structured Matrix: {train_feats['X_struct'].shape} (dense)")
    print(f"     TF-IDF Matrix:     {train_feats['X_tfidf'].shape} (sparse)")
    print(f"     Combined Matrix:   {train_feats['X_all'].shape} (sparse)")

    # 3. Feature Selection Experiment (SelectKBest)
    print("\n[STEP 4] Running Feature Selection Experiment (SelectKBest, k=500)...")
    all_feature_names = (
        pipeline.vectorizer.get_feature_names_out().tolist() +
        pipeline.structured_feature_names
    )
    selector, ranking_df = feature_selection.run_feature_selection_experiment(
        train_feats["X_all"],
        train_feats["y_clf"],
        all_feature_names,
        k=500,
        save_outputs=True
    )

    # Apply feature selector to create X_selected for all partitions
    print("[FEATURE SELECTION] Transforming partitions to X_selected (500 features)...")
    train_feats["X_selected"] = selector.transform(train_feats["X_all"])
    val_feats["X_selected"] = selector.transform(val_feats["X_all"])
    test_feats["X_selected"] = selector.transform(test_feats["X_all"])
    print(f"  -> Selected Feature Matrix Shape: {train_feats['X_selected'].shape}")

    # 4. Train Classifiers with Validation-First Protocol
    print("\n[STEP 5] Training & Benchmarking Classifiers across All-Feature vs. Selected-Feature Experiments...")
    trained_clfs, clf_results_df, test_preds = train_classifiers.train_and_evaluate_classifiers(
        train_feats,
        val_feats,
        test_feats,
        save_models=True
    )

    # Save separate experiment CSVs
    all_feat_df = clf_results_df[clf_results_df["Feature_Set"].str.contains("All Features|Structured|TF-IDF")].drop(columns=["Confusion_Matrix"])
    selected_feat_df = clf_results_df[clf_results_df["Feature_Set"].str.contains("Selected Features|Structured|TF-IDF")].drop(columns=["Confusion_Matrix"])
    
    all_feat_csv = config.RESULTS_DIR / "classification_all_features_comparison.csv"
    selected_feat_csv = config.RESULTS_DIR / "classification_selected_features_comparison.csv"
    master_clf_csv = config.RESULTS_DIR / "classification_model_comparison.csv"
    
    all_feat_df.to_csv(all_feat_csv, index=False)
    selected_feat_df.to_csv(selected_feat_csv, index=False)
    clf_results_df.drop(columns=["Confusion_Matrix"]).to_csv(master_clf_csv, index=False)
    
    print(f"\n[OUTPUT SAVED] All-features comparison: {all_feat_csv}")
    print(f"[OUTPUT SAVED] Selected-features comparison: {selected_feat_csv}")
    print(f"[OUTPUT SAVED] Master comparison: {master_clf_csv}")

    # Generate Classifier Diagnostic Visualizations
    target_classes = ["weak_positive", "medium_positive", "strong_positive"]
    plot_confusion_matrices(clf_results_df, target_classes)
    plot_model_comparison_bars(clf_results_df)

    # 5. Train Regressors with Regularization Shrinkage Audit
    print("\n[STEP 6] Training Linear Regressors with Multi-Alpha Regularization Audit...")
    trained_regs, reg_results_df, reg_preds = train_regressor.train_and_evaluate_regressors(
        train_feats,
        val_feats,
        test_feats,
        save_models=True
    )

    # 6. Final Report Summary
    print("\n" + "=" * 80)
    print("                    VALIDATION LEADERBOARD (MODEL SELECTION)             ")
    print("=" * 80)
    val_sorted = clf_results_df.sort_values(by="Val_Macro_F1", ascending=False)
    print(val_sorted[[
        "Model", "Feature_Set", "Val_Accuracy", "Val_Macro_F1", "Val_Weighted_F1"
    ]].to_string(index=False))

    print("\n" + "=" * 80)
    print("                    FINAL HELD-OUT TEST BENCHMARK                       ")
    print("=" * 80)
    test_sorted = clf_results_df.sort_values(by="Val_Macro_F1", ascending=False)
    print(test_sorted[[
        "Model", "Feature_Set", "Test_Accuracy", "Test_Macro_F1", "Test_Weighted_F1",
        "Test_F1_Weak", "Test_F1_Medium", "Test_F1_Strong", "Fit_Time_Sec"
    ]].to_string(index=False))

    best_val_model = val_sorted.iloc[0]["Model"]
    best_val_f1 = val_sorted.iloc[0]["Val_Macro_F1"]
    best_test_f1 = val_sorted.iloc[0]["Test_Macro_F1"]
    print(f"\n[MODEL SELECTION DECISION]: Winning model chosen on Validation set is '{best_val_model}' (Val Macro F1: {best_val_f1:.4f} -> Final Test Macro F1: {best_test_f1:.4f})")


if __name__ == "__main__":
    main()
