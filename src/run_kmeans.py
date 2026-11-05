"""
run_kmeans.py - Experiment 11: K-Means Clustering on Resume-Job Compatibility Features

Academic ML Lab Concepts Demonstrated:
1. Unsupervised Partitioning via Lloyd's Algorithm:
   - Iteratively assigns points to nearest centroid: argmin_k ||x_i - mu_k||^2
   - Updates centroids as mean of assigned samples: mu_k = (1 / |S_k|) * sum_{x in S_k} x
2. Determining Optimal Clusters:
   - Elbow Method: Within-Cluster Sum of Squares (Inertia) vs Number of Clusters k.
   - Silhouette Coefficient: s(i) = (b(i) - a(i)) / max(a(i), b(i)), evaluating cohesion and separation.
   - Subsampling strategy (5,000 points) to compute pairwise silhouette distances efficiently on CPU.
3. Crucial Pedagogical Principle:
   - Clusters are strictly unsupervised groupings in feature space (grouping by domain, experience, etc.).
   - Clusters do NOT inherently correspond to supervised compatibility labels (weak/medium/strong).
   - Evaluated via Contingency Cross-tabulation, Adjusted Rand Index (ARI), and Normalized Mutual Info (NMI).
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)

import config
import feature_engineering


def run_kmeans_experiments(
    train_feats: Dict[str, Any],
    val_feats: Dict[str, Any],
    k_range: range = range(2, 9),
    chosen_k: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes K-Means elbow analysis, silhouette computation, and cluster characterization.
    
    Args:
        train_feats, val_feats: Feature dictionaries containing X_struct, y_clf.
        k_range: Range of cluster counts to test.
        chosen_k: Selected cluster count for detailed visualization and profiling.
        random_state: Reproducibility seed.
        
    Returns:
        Dictionary of cluster model, metrics, and diagnostics.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: K-MEANS CLUSTERING (UNSUPERVISED COMPATIBILITY PROFILING)")
    print("=" * 70)

    X_train = train_feats["X_struct"]
    y_train = train_feats["y_clf"]
    X_val = val_feats["X_struct"]
    y_val = val_feats["y_clf"]
    feature_names = train_feats.get("feature_names_struct", [f"feat_{i}" for i in range(X_train.shape[1])])

    print(f"[DATA] Train samples: {X_train.shape[0]:,} | Features: {X_train.shape[1]}")
    print("Pedagogical Note:")
    print("  K-Means is unsupervised and groups candidate-job pairs solely by spatial proximity")
    print("  in standardized feature space. It has no access to ground-truth labels during training.")

    # 1. Elbow Method & Silhouette Analysis across k
    print(f"\n[ANALYSIS] Evaluating K-Means for k in {list(k_range)}...")
    inertias = []
    silhouette_scores_list = []
    k_values = list(k_range)
    
    # Subsample 5,000 samples for CPU-friendly silhouette distance matrix computation
    np.random.seed(random_state)
    subsample_idx = np.random.choice(X_train.shape[0], size=min(5000, X_train.shape[0]), replace=False)
    X_sub = X_train[subsample_idx]

    for k in k_values:
        t0 = time.time()
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=random_state)
        km.fit(X_train)
        inertias.append(km.inertia_)
        
        # Silhouette computed on sample
        sub_labels = km.predict(X_sub)
        sil = silhouette_score(X_sub, sub_labels)
        silhouette_scores_list.append(sil)
        print(f"  k = {k:2d} | Inertia = {km.inertia_:12.1f} | Silhouette Score = {sil:.4f} | Time: {time.time()-t0:.2f}s")

    # 2. Fit Chosen Model (k = 3 for comparison with 3 compatibility labels)
    print(f"\n[CLUSTERING] Fitting primary K-Means model with k={chosen_k} clusters...")
    km_chosen = KMeans(n_clusters=chosen_k, init="k-means++", n_init=15, max_iter=300, random_state=random_state)
    km_chosen.fit(X_train)
    train_clusters = km_chosen.labels_
    val_clusters = km_chosen.predict(X_val)

    # 3. Cluster Contingency vs Supervised Ground Truth
    print("\n[CONTINGENCY] Comparing Unsupervised Clusters to Supervised 'label_strength':")
    df_contingency = pd.crosstab(
        pd.Series(y_val, name="Ground Truth (label_strength)"),
        pd.Series(val_clusters, name=f"K-Means Cluster (k={chosen_k})"),
        margins=True,
    )
    print(df_contingency)

    # Compute Clustering Alignment Metrics
    ari = adjusted_rand_score(y_val, val_clusters)
    nmi = normalized_mutual_info_score(y_val, val_clusters)
    homo = homogeneity_score(y_val, val_clusters)
    print("\nClustering vs Ground Truth Agreement Metrics:")
    print(f"  Adjusted Rand Index (ARI):        {ari:.4f} (scale -1 to 1; ~0 denotes chance alignment)")
    print(f"  Normalized Mutual Info (NMI):     {nmi:.4f} (scale 0 to 1; measures shared information)")
    print(f"  Homogeneity Score:                {homo:.4f} (scale 0 to 1; measures single-class purity)")
    print("  -> Conclusion: Demonstrates that unsupervised spatial clustering identifies structural patterns")
    print("     (such as high experience or IT specialization) but cannot replace supervised compatibility labels.")

    # 4. Profile Cluster Centroids in Original Feature Space
    df_centroids = pd.DataFrame(km_chosen.cluster_centers_, columns=feature_names)
    df_centroids.index = [f"Cluster_{i}" for i in range(chosen_k)]
    print("\nCluster Centroids (Z-score standardized coordinates):")
    top_distinguishing = ["tfidf_similarity", "jaccard_skills", "years_experience", "years_required", "resume_word_count"]
    available_cols = [c for c in top_distinguishing if c in df_centroids.columns]
    print(df_centroids[available_cols].round(3))

    # 5. Compile Summary DataFrames
    df_elbow = pd.DataFrame({
        "k": k_values,
        "inertia": [round(x, 2) for x in inertias],
        "silhouette_score": [round(x, 4) for x in silhouette_scores_list],
    })

    df_summary = pd.DataFrame([{
        "model": f"K-Means (k={chosen_k})",
        "n_clusters": chosen_k,
        "inertia": round(float(km_chosen.inertia_), 2),
        "validation_silhouette": round(float(silhouette_score(X_val[:5000], val_clusters[:5000])), 4),
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "homogeneity": round(homo, 4),
    }])

    # 6. Save Artifacts & Results
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    model_path = config.MODELS_DIR / "kmeans_clusterer.joblib"
    joblib.dump(km_chosen, model_path)
    print(f"\n[MODEL SAVED] {model_path}")

    csv_path = config.RESULTS_DIR / "kmeans_clustering_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"[RESULTS SAVED] {csv_path}")

    contingency_path = config.RESULTS_DIR / "kmeans_contingency_table.csv"
    df_contingency.to_csv(contingency_path)
    print(f"[CONTINGENCY SAVED] {contingency_path}")

    elbow_path = config.RESULTS_DIR / "kmeans_elbow_metrics.csv"
    df_elbow.to_csv(elbow_path, index=False)

    # 7. Generate Diagnostic Visualizations
    plot_kmeans_diagnostics(k_values, inertias, silhouette_scores_list, km_chosen, X_val, val_clusters)

    return {
        "model": km_chosen,
        "df_elbow": df_elbow,
        "df_summary": df_summary,
        "df_contingency": df_contingency,
    }


def plot_kmeans_diagnostics(
    k_values: list,
    inertias: list,
    silhouette_scores: list,
    km_model: KMeans,
    X_val: np.ndarray,
    val_clusters: np.ndarray,
):
    """Generates dual-axis elbow/silhouette plot and 2D PCA cluster projection."""
    # Plot 1: Elbow Curve (Inertia) and Silhouette Scores
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_in = "#1f77b4"
    ax1.plot(k_values, inertias, color=color_in, marker="o", lw=2.2, label="Inertia (Within-Cluster Sum of Squares)")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Inertia", color=color_in, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_in)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_sil = "#2ca02c"
    ax2.plot(k_values, silhouette_scores, color=color_sil, marker="s", lw=2, linestyle="--", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score (Cohesion / Separation)", color=color_sil, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_sil)

    plt.title("K-Means Diagnostic: Elbow Method (Inertia) & Silhouette Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    elbow_plot_path = config.PLOTS_DIR / "kmeans_elbow_analysis.png"
    plt.savefig(elbow_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT SAVED] {elbow_plot_path}")

    # Plot 2: 2D PCA Cluster Projection
    pca = PCA(n_components=2, random_state=42)
    X_val_2d = pca.fit_transform(X_val)
    centroids_2d = pca.transform(km_model.cluster_centers_)

    subsample_idx = np.random.RandomState(42).choice(len(X_val), size=min(4000, len(X_val)), replace=False)
    df_proj = pd.DataFrame({
        "PC1": X_val_2d[subsample_idx, 0],
        "PC2": X_val_2d[subsample_idx, 1],
        "Cluster": [f"Cluster {c}" for c in val_clusters[subsample_idx]],
    })

    plt.figure(figsize=(9, 7))
    palette = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    for c_idx in range(km_model.n_clusters):
        cluster_name = f"Cluster {c_idx}"
        subset = df_proj[df_proj["Cluster"] == cluster_name]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            c=palette[c_idx % len(palette)],
            label=cluster_name,
            alpha=0.6,
            s=30,
            edgecolors="none",
        )
    plt.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="black",
        s=200,
        marker="X",
        label="Cluster Centroids",
        edgecolors="white",
        linewidths=1.5,
    )
    plt.title(f"K-Means 2D Projection via Principal Components (k={km_model.n_clusters})", fontsize=12, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=10)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Assigned Cluster", loc="best")
    plt.tight_layout()
    proj_plot_path = config.PLOTS_DIR / "kmeans_cluster_projection.png"
    plt.savefig(proj_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT SAVED] {proj_plot_path}")


def main():
    print("[INIT] Loading preprocessed candidate-grouped features...")
    train_feats, val_feats, test_feats = feature_engineering.load_or_create_features()
    run_kmeans_experiments(train_feats, val_feats, chosen_k=3)
    print("\n[DONE] Experiment 11: K-Means Clustering completed successfully.")


if __name__ == "__main__":
    main()
