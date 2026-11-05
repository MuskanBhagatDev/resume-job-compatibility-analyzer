"""
run_pca.py - Experiment 6: Dimensionality Reduction via PCA and TruncatedSVD (LSA)

Academic ML Lab Concepts Demonstrated:
1. Principal Component Analysis (PCA):
   - Computes orthogonal eigenvectors of the sample covariance matrix.
   - Requires explicit mean centering: X_centered = X - mu.
   - Applied to dense structured features (16 dimensions -> 2D visualization).
2. Truncated Singular Value Decomposition (TruncatedSVD / LSA):
   - Computes X approx U_k * Sigma_k * V_k^T directly on sparse matrices without mean centering.
   - Crucial distinction: Centering a sparse 85,392 x 1,500 TF-IDF matrix destroys sparsity,
     causing massive RAM consumption (>1 GB) and memory errors on standard laptop CPUs.
   - TruncatedSVD performs Latent Semantic Analysis (LSA) directly on CSR sparse matrices in seconds.
3. Downstream Utility: Evaluates classification performance (Logistic Regression) on
   50 SVD semantic components vs raw 1,500 sparse features.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import config
import feature_engineering


def run_pca_and_svd_experiments(
    train_feats: Dict[str, Any],
    val_feats: Dict[str, Any],
    test_feats: Dict[str, Any],
    n_svd_components: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes PCA on dense features and TruncatedSVD on sparse TF-IDF matrices.
    
    Args:
        train_feats, val_feats, test_feats: Feature dictionaries.
        n_svd_components: Number of latent semantic components to extract from TF-IDF.
        random_state: Reproducibility seed.
        
    Returns:
        Dictionary of fitted transformers, projections, and evaluation summary.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: DIMENSIONALITY REDUCTION (PCA & TRUNCATED SVD)")
    print("=" * 70)

    X_train_struct = train_feats["X_struct"]
    X_val_struct = val_feats["X_struct"]
    X_test_struct = test_feats["X_struct"]

    X_train_tfidf = train_feats["X_tfidf"]
    X_val_tfidf = val_feats["X_tfidf"]
    X_test_tfidf = test_feats["X_tfidf"]

    y_train = train_feats["y_clf"]
    y_val = val_feats["y_clf"]
    y_test = test_feats["y_clf"]

    # -------------------------------------------------------------
    # PART 1: TruncatedSVD (LSA) on Sparse TF-IDF (1,500 -> 50 dims)
    # -------------------------------------------------------------
    print("\n[PART 1: TruncatedSVD on Sparse TF-IDF]")
    print(f"  Input matrix: Sparse CSR {X_train_tfidf.shape} (NNZ: {X_train_tfidf.nnz:,})")
    print("  Theoretical Note:")
    print("    Standard PCA requires mean-centering (X - mu). On a sparse 85,392 x 1,500 matrix,")
    print("    centering fills every zero entry, destroying sparsity and consuming ~1 GB RAM.")
    print("    TruncatedSVD operates directly on CSR matrices without centering, enabling fast LSA.")

    t0 = time.time()
    svd = TruncatedSVD(n_components=n_svd_components, algorithm="randomized", random_state=random_state)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    svd_fit_time = time.time() - t0

    X_val_svd = svd.transform(X_val_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)

    total_var_explained = svd.explained_variance_ratio_.sum()
    print(f"  -> TruncatedSVD fit in {svd_fit_time:.2f}s")
    print(f"  -> Reduced dimensions: 1,500 TF-IDF features -> {n_svd_components} latent semantic components")
    print(f"  -> Cumulative explained variance (top {n_svd_components}): {total_var_explained:.4f} ({total_var_explained*100:.2f}%)")

    # -------------------------------------------------------------
    # PART 2: Standard PCA on Dense Structured Features (16 -> 2 dims)
    # -------------------------------------------------------------
    print("\n[PART 2: Standard PCA on Dense Structured Features]")
    print(f"  Input matrix: Dense array {X_train_struct.shape}")
    print("  Applies mean-centering and orthogonal projection into 2 Principal Components.")

    t0 = time.time()
    pca_struct = PCA(n_components=2, random_state=random_state)
    X_train_pca2 = pca_struct.fit_transform(X_train_struct)
    pca_fit_time = time.time() - t0

    X_val_pca2 = pca_struct.transform(X_val_struct)
    pca_var_ratios = pca_struct.explained_variance_ratio_
    print(f"  -> PCA fit in {pca_fit_time:.3f}s")
    print(f"  -> PC1 Explained Variance: {pca_var_ratios[0]:.4f} ({pca_var_ratios[0]*100:.2f}%)")
    print(f"  -> PC2 Explained Variance: {pca_var_ratios[1]:.4f} ({pca_var_ratios[1]*100:.2f}%)")
    print(f"  -> Total 2D Explained Variance: {pca_var_ratios.sum():.4f} ({pca_var_ratios.sum()*100:.2f}%)")

    # -------------------------------------------------------------
    # PART 3: Downstream Classification Comparison (Logistic Regression)
    # -------------------------------------------------------------
    print("\n[PART 3: Downstream Classification Benchmark]")
    print("  Evaluating Logistic Regression on:")
    print("    A. Full Sparse TF-IDF (1,500 features)")
    print(f"    B. TruncatedSVD Semantic Components ({n_svd_components} features)")
    print("    C. Dense Structured Compatibility Features (16 features)")

    # Model A: Raw TF-IDF
    lr_tfidf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=random_state)
    lr_tfidf.fit(X_train_tfidf, y_train)
    val_pred_tfidf = lr_tfidf.predict(X_val_tfidf)
    val_f1_tfidf = f1_score(y_val, val_pred_tfidf, average="macro", zero_division=0)
    val_acc_tfidf = accuracy_score(y_val, val_pred_tfidf)

    # Model B: TruncatedSVD Components
    lr_svd = LogisticRegression(max_iter=200, class_weight="balanced", random_state=random_state)
    lr_svd.fit(X_train_svd, y_train)
    val_pred_svd = lr_svd.predict(X_val_svd)
    val_f1_svd = f1_score(y_val, val_pred_svd, average="macro", zero_division=0)
    val_acc_svd = accuracy_score(y_val, val_pred_svd)

    # Model C: Structured Features
    lr_struct = LogisticRegression(max_iter=200, class_weight="balanced", random_state=random_state)
    lr_struct.fit(X_train_struct, y_train)
    val_pred_struct = lr_struct.predict(X_val_struct)
    val_f1_struct = f1_score(y_val, val_pred_struct, average="macro", zero_division=0)
    val_acc_struct = accuracy_score(y_val, val_pred_struct)

    # Evaluate Winner on Test Set
    test_pred_svd = lr_svd.predict(X_test_svd)
    test_f1_svd = f1_score(y_test, test_pred_svd, average="macro", zero_division=0)
    test_acc_svd = accuracy_score(y_test, test_pred_svd)

    print(f"  Results on Validation Set:")
    print(f"    1. Full TF-IDF (1,500 feats)   : Macro F1 = {val_f1_tfidf:.4f} | Accuracy = {val_acc_tfidf*100:.2f}%")
    print(f"    2. TruncatedSVD ({n_svd_components} feats)   : Macro F1 = {val_f1_svd:.4f} | Accuracy = {val_acc_svd*100:.2f}%")
    print(f"    3. Structured Only (16 feats)  : Macro F1 = {val_f1_struct:.4f} | Accuracy = {val_acc_struct*100:.2f}%")
    print(f"  TruncatedSVD Held-out Test Set   : Macro F1 = {test_f1_svd:.4f} | Accuracy = {test_acc_svd*100:.2f}%")

    # 4. Save Summary CSV
    comparison_records = [
        {
            "representation": "Full TF-IDF",
            "dimensionality": 1500,
            "reduction_method": "None",
            "val_accuracy": round(val_acc_tfidf, 4),
            "val_macro_f1": round(val_f1_tfidf, 4),
            "test_accuracy": "-",
            "test_macro_f1": "-",
            "compression_ratio": "1.0x (Baseline)",
        },
        {
            "representation": f"TruncatedSVD Components",
            "dimensionality": n_svd_components,
            "reduction_method": "TruncatedSVD (Randomized)",
            "val_accuracy": round(val_acc_svd, 4),
            "val_macro_f1": round(val_f1_svd, 4),
            "test_accuracy": round(test_acc_svd, 4),
            "test_macro_f1": round(test_f1_svd, 4),
            "compression_ratio": f"{1500 / n_svd_components:.1f}x reduction",
        },
        {
            "representation": "Engineered Structured",
            "dimensionality": 16,
            "reduction_method": "Domain Feature Engineering",
            "val_accuracy": round(val_acc_struct, 4),
            "val_macro_f1": round(val_f1_struct, 4),
            "test_accuracy": "-",
            "test_macro_f1": "-",
            "compression_ratio": f"{1500 / 16:.1f}x reduction",
        },
    ]
    df_pca_summary = pd.DataFrame(comparison_records)
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    csv_path = config.RESULTS_DIR / "pca_svd_summary.csv"
    df_pca_summary.to_csv(csv_path, index=False)
    print(f"\n[RESULTS SAVED] {csv_path}")

    # 5. Save model artifacts
    artifacts = {
        "truncated_svd": svd,
        "pca_struct": pca_struct,
        "lr_svd_model": lr_svd,
    }
    model_path = config.MODELS_DIR / "pca_svd_reducers.joblib"
    joblib.dump(artifacts, model_path)
    print(f"[REDUCERS SAVED] {model_path}")

    # 6. Generate Diagnostic Plots
    plot_pca_and_svd_diagnostics(svd, X_val_pca2, y_val, pca_var_ratios)

    return {
        "svd": svd,
        "pca": pca_struct,
        "summary_df": df_pca_summary,
        "X_train_pca2": X_train_pca2,
        "X_val_pca2": X_val_pca2,
    }


def plot_pca_and_svd_diagnostics(
    svd: TruncatedSVD,
    X_val_pca2: np.ndarray,
    y_val: np.ndarray,
    pca_var_ratios: np.ndarray,
):
    """Generates Scree Plot for TruncatedSVD and 2D Scatter for Structured PCA."""
    # Plot 1: SVD Scree & Cumulative Variance
    cum_var = np.cumsum(svd.explained_variance_ratio_)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(range(1, len(svd.explained_variance_ratio_) + 1), svd.explained_variance_ratio_, alpha=0.6, color="#4c72b0", label="Individual Variance")
    ax1.set_xlabel("Latent Semantic Component Index", fontsize=10)
    ax1.set_ylabel("Individual Explained Variance Ratio", color="#4c72b0", fontsize=10)
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cum_var) + 1), cum_var, color="#c44e52", lw=2.2, marker="o", markersize=3, label="Cumulative Variance")
    ax2.set_ylabel("Cumulative Explained Variance Ratio", color="#c44e52", fontsize=10)
    ax2.set_ylim(0, max(cum_var) * 1.15)
    
    plt.title(f"TruncatedSVD (LSA) Scree & Cumulative Variance Plot ({len(cum_var)} Components)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    scree_path = config.PLOTS_DIR / "pca_svd_scree.png"
    plt.savefig(scree_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT SAVED] {scree_path}")

    # Plot 2: 2D PCA Projection of Structured Features
    plt.figure(figsize=(9, 7))
    # Subsample 4000 points from validation set for clean scatter rendering
    sample_indices = np.random.RandomState(42).choice(len(y_val), size=min(4000, len(y_val)), replace=False)
    df_scatter = pd.DataFrame({
        "PC1": X_val_pca2[sample_indices, 0],
        "PC2": X_val_pca2[sample_indices, 1],
        "Label": y_val[sample_indices],
    })
    palette = {"weak_positive": "#e74c3c", "medium_positive": "#3498db", "strong_positive": "#2ecc71"}
    for label, color in palette.items():
        subset = df_scatter[df_scatter["Label"] == label]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            c=color,
            label=label,
            alpha=0.6,
            s=25,
            edgecolors="none",
        )
    plt.title(f"2D PCA Projection of Structured Features (PC1: {pca_var_ratios[0]*100:.1f}%, PC2: {pca_var_ratios[1]*100:.1f}%)",
              fontsize=12, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca_var_ratios[0]*100:.1f}%)", fontsize=10)
    plt.ylabel(f"Principal Component 2 ({pca_var_ratios[1]*100:.1f}%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Compatibility Label", loc="best")
    plt.tight_layout()
    scatter_path = config.PLOTS_DIR / "pca_struct_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT SAVED] {scatter_path}")


def main():
    print("[INIT] Loading preprocessed candidate-grouped features...")
    train_feats, val_feats, test_feats = feature_engineering.load_or_create_features()
    run_pca_and_svd_experiments(train_feats, val_feats, test_feats)
    print("\n[DONE] Experiment 6: PCA & TruncatedSVD completed successfully.")


if __name__ == "__main__":
    main()
