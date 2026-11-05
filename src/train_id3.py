"""
train_id3.py - Experiment 7: Entropy / Information-Gain Decision Tree (ID3 Variant)

Academic ML Lab Concepts Demonstrated:
1. Shannon Entropy: H(S) = - sum_{c} p_c * log2(p_c), measuring class impurity.
2. Information Gain: IG(S, A) = H(S) - sum_v (|S_v| / |S|) * H(S_v), measuring
   the reduction in uncertainty after splitting on attribute A.
3. Methodological Note (Textbook ID3 vs Scikit-Learn CART):
   - Textbook ID3 (Quinlan, 1986) creates multi-way categorical branches and does not natively
     handle continuous numerical features.
   - Scikit-learn's DecisionTreeClassifier utilizes an optimized CART formulation that
     supports continuous features by finding binary split thresholds (x <= threshold)
     that maximize Shannon Information Gain when criterion='entropy' is specified.
4. Decision Tree Interpretation: Visual tree layout and Mean Decrease in Impurity (MDI).
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

import config
import feature_engineering


def train_id3_decision_tree(
    train_feats: Dict[str, Any],
    val_feats: Dict[str, Any],
    test_feats: Dict[str, Any],
    max_depth: int = 4,
    min_samples_split: int = 50,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and visualizes an Entropy/Information-Gain Decision Tree on structured features.
    
    Args:
        train_feats, val_feats, test_feats: Feature dictionaries containing X_struct, y_clf.
        max_depth: Depth limit for interpretability and overfitting prevention.
        min_samples_split, min_samples_leaf: Regularization constraints.
        random_state: Seed for reproducibility.
        
    Returns:
        Dictionary containing trained model, metrics DataFrame, and importances.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: ENTROPY / INFORMATION-GAIN DECISION TREE (ID3 VARIANT)")
    print("=" * 70)
    print("Methodological Note:")
    print("  Using criterion='entropy' (Shannon Information Gain) with CART binary partitioning.")
    print("  Interpretable depth limit: max_depth=4 to allow full pedagogical inspection.")

    X_train = train_feats["X_struct"]
    y_train = train_feats["y_clf"]
    X_val = val_feats["X_struct"]
    y_val = val_feats["y_clf"]
    X_test = test_feats["X_struct"]
    y_test = test_feats["y_clf"]
    feature_names = train_feats.get("feature_names_struct", [f"feat_{i}" for i in range(X_train.shape[1])])
    classes = sorted(list(set(y_train)))

    print(f"[DATA] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"[FEATURES] {len(feature_names)} structured & compatibility attributes")

    # 1. Instantiate Entropy Decision Tree with class weighting
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )

    print("\n[TRAINING] Fitting Entropy-based Decision Tree...")
    t0 = time.time()
    tree.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[TRAINING COMPLETE] Time elapsed: {train_time:.3f}s")
    print(f"  -> Total leaf nodes: {tree.get_n_leaves()}")
    print(f"  -> Tree depth: {tree.get_depth()}")

    # 2. Evaluate on Validation Set
    val_pred = tree.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_macro = f1_score(y_val, val_pred, average="macro", zero_division=0)
    val_weighted = f1_score(y_val, val_pred, average="weighted", zero_division=0)

    # 3. Evaluate on Held-out Test Set
    test_pred = tree.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_macro = f1_score(y_test, test_pred, average="macro", zero_division=0)
    test_weighted = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    print("\n" + "-" * 50)
    print("VALIDATION SET PERFORMANCE:")
    print(f"  Accuracy:    {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Macro F1:    {val_macro:.4f}")
    print(f"  Weighted F1: {val_weighted:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred, target_names=classes, digits=4, zero_division=0))

    print("-" * 50)
    print("HELD-OUT TEST SET PERFORMANCE:")
    print(f"  Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Macro F1:    {test_macro:.4f}")
    print(f"  Weighted F1: {test_weighted:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, target_names=classes, digits=4, zero_division=0))

    # 4. Feature Importances (Information Gain / MDI)
    df_importances = pd.DataFrame({
        "feature": feature_names,
        "importance_entropy": tree.feature_importances_
    }).sort_values(by="importance_entropy", ascending=False).reset_index(drop=True)
    
    print("\nTop 8 Most Informative Features (Information Gain / Entropy Impurity Reduction):")
    for idx, row in df_importances.head(8).iterrows():
        print(f"  {idx+1}. {row['feature']:<25} : {row['importance_entropy']:.4f}")

    # 5. Compile metrics DataFrame
    metrics_records = [
        {
            "model": "ID3 Decision Tree (Entropy)",
            "criterion": "entropy",
            "max_depth": max_depth,
            "n_leaves": tree.get_n_leaves(),
            "train_time_sec": round(train_time, 3),
            "split": "validation",
            "accuracy": round(val_acc, 4),
            "macro_f1": round(val_macro, 4),
            "weighted_f1": round(val_weighted, 4),
            "macro_precision": round(precision_score(y_val, val_pred, average="macro", zero_division=0), 4),
            "macro_recall": round(recall_score(y_val, val_pred, average="macro", zero_division=0), 4),
        },
        {
            "model": "ID3 Decision Tree (Entropy)",
            "criterion": "entropy",
            "max_depth": max_depth,
            "n_leaves": tree.get_n_leaves(),
            "train_time_sec": round(train_time, 3),
            "split": "test",
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_macro, 4),
            "weighted_f1": round(test_weighted, 4),
            "macro_precision": round(precision_score(y_test, test_pred, average="macro", zero_division=0), 4),
            "macro_recall": round(recall_score(y_test, test_pred, average="macro", zero_division=0), 4),
        },
    ]
    df_metrics = pd.DataFrame(metrics_records)

    # 6. Save model and results
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    model_path = config.MODELS_DIR / "id3_decision_tree.joblib"
    joblib.dump(tree, model_path)
    print(f"\n[MODEL SAVED] {model_path}")

    csv_path = config.RESULTS_DIR / "id3_decision_tree_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"[RESULTS SAVED] {csv_path}")

    imp_path = config.RESULTS_DIR / "id3_feature_importances.csv"
    df_importances.to_csv(imp_path, index=False)
    print(f"[IMPORTANCES SAVED] {imp_path}")

    # 7. Visualizations: Tree Diagram & Feature Importance Bar Chart
    plot_decision_tree_and_importance(tree, feature_names, classes, df_importances)

    return {
        "model": tree,
        "metrics_df": df_metrics,
        "importances_df": df_importances,
    }


def plot_decision_tree_and_importance(
    tree: DecisionTreeClassifier,
    feature_names: list,
    classes: list,
    df_importances: pd.DataFrame,
):
    """Generates visual decision tree diagram and feature importance plot."""
    # Plot 1: Visual Decision Tree
    plt.figure(figsize=(24, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=8,
        precision=2,
        proportion=True,
    )
    plt.title("Entropy-Based Decision Tree (Information Gain Splits, Max Depth = 4)", fontsize=16, fontweight="bold")
    tree_path = config.PLOTS_DIR / "id3_decision_tree.png"
    plt.savefig(tree_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT SAVED] {tree_path}")

    # Plot 2: Feature Importances
    plt.figure(figsize=(10, 6))
    df_sorted = df_importances.sort_values(by="importance_entropy", ascending=True)
    plt.barh(df_sorted["feature"], df_sorted["importance_entropy"], color="#2b5c8f")
    plt.title("Feature Importance via Shannon Entropy Impurity Reduction", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Decrease in Entropy (Importance)", fontsize=10)
    plt.ylabel("Engineered Feature", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    imp_plot_path = config.PLOTS_DIR / "id3_feature_importance.png"
    plt.savefig(imp_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT SAVED] {imp_plot_path}")


def main():
    print("[INIT] Loading preprocessed candidate-grouped features...")
    train_feats, val_feats, test_feats = feature_engineering.load_or_create_features()
    train_id3_decision_tree(train_feats, val_feats, test_feats)
    print("\n[DONE] Experiment 7: ID3 Decision Tree completed successfully.")


if __name__ == "__main__":
    main()
