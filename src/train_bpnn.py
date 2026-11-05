"""
train_bpnn.py - Experiment 5: Backpropagation Multi-Layer Perceptron (MLP)

Implementation of a CPU-friendly Artificial Neural Network trained using
Error Backpropagation with Adam optimizer on genuine candidate-job features.

Academic ML Lab Concepts Demonstrated:
1. Multi-Layer Perceptron (MLP) Architecture: Input -> Hidden Layers -> Softmax Output.
2. Error Backpropagation Algorithm: Forward pass to compute activations,
   cross-entropy loss calculation, and backward pass via chain rule to compute
   weight gradients (w_{ij} <- w_{ij} - eta * dE/dw_{ij}).
3. Handling severe class imbalance via sample weighting in mini-batch backpropagation.
4. Validation loss tracking with early stopping to prevent overfitting.
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
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

import config
import feature_engineering


def train_bpnn_model(
    train_feats: Dict[str, Any],
    val_feats: Dict[str, Any],
    test_feats: Dict[str, Any],
    hidden_layer_sizes: tuple = (64, 32),
    max_iter: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and evaluates a Backpropagation Neural Network (MLPClassifier).
    
    Args:
        train_feats, val_feats, test_feats: Feature dictionaries containing X_struct, y_clf.
        hidden_layer_sizes: Tuple representing neurons per hidden layer.
        max_iter: Maximum training iterations (epochs).
        random_state: Seed for reproducibility.
        
    Returns:
        Dictionary containing trained model, metrics DataFrame, and predictions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: BACKPROPAGATION MULTI-LAYER PERCEPTRON (MLP)")
    print("=" * 70)
    
    X_train = train_feats["X_struct"]
    y_train = train_feats["y_clf"]
    X_val = val_feats["X_struct"]
    y_val = val_feats["y_clf"]
    X_test = test_feats["X_struct"]
    y_test = test_feats["y_clf"]
    feature_names = train_feats.get("feature_names_struct", [])
    
    print(f"[DATA] Train shape: {X_train.shape} | Val shape: {X_val.shape} | Test shape: {X_test.shape}")
    print(f"[FEATURES] Input dimensions: {X_train.shape[1]} structured/similarity features")
    print(f"[ARCHITECTURE] Input ({X_train.shape[1]}) -> Dense({hidden_layer_sizes[0]}, ReLU) "
          f"-> Dense({hidden_layer_sizes[1]}, ReLU) -> Output (3 classes, Softmax)")

    # 1. Compute sample weights to balance gradients across imbalanced classes
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    
    # 2. Instantiate CPU-friendly MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.001,             # L2 regularization penalty
        batch_size=256,          # Mini-batch gradient descent
        learning_rate="adaptive",
        learning_rate_init=0.005,
        max_iter=max_iter,
        early_stopping=True,     # Monitor validation loss
        n_iter_no_change=10,     # Stop if no improvement for 10 epochs
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
    )
    
    print("\n[TRAINING] Training Backpropagation Neural Network via Adam...")
    t0 = time.time()
    mlp.fit(X_train, y_train, sample_weight=sample_weights)
    train_time = time.time() - t0
    
    print(f"[TRAINING COMPLETE] Time elapsed: {train_time:.2f}s")
    print(f"  -> Converged after: {mlp.n_iter_} iterations")
    print(f"  -> Final training loss: {mlp.loss_:.4f}")
    if hasattr(mlp, "best_loss_") and mlp.best_loss_ is not None:
        print(f"  -> Best internal validation loss: {mlp.best_loss_:.4f}")

    # 3. Evaluate on Validation Set
    val_pred = mlp.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_macro = f1_score(y_val, val_pred, average="macro", zero_division=0)
    val_weighted = f1_score(y_val, val_pred, average="weighted", zero_division=0)

    # 4. Evaluate on Held-out Test Set
    test_pred = mlp.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_macro = f1_score(y_test, test_pred, average="macro", zero_division=0)
    test_weighted = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    print("\n" + "-" * 50)
    print("VALIDATION SET PERFORMANCE:")
    print(f"  Accuracy:    {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Macro F1:    {val_macro:.4f}")
    print(f"  Weighted F1: {val_weighted:.4f}")
    print("\nValidation Classification Report:")
    classes = sorted(list(set(y_train)))
    print(classification_report(y_val, val_pred, target_names=classes, digits=4, zero_division=0))

    print("-" * 50)
    print("HELD-OUT TEST SET PERFORMANCE:")
    print(f"  Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Macro F1:    {test_macro:.4f}")
    print(f"  Weighted F1: {test_weighted:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, target_names=classes, digits=4, zero_division=0))

    # 5. Compile metrics DataFrame
    metrics_records = [
        {
            "model": "Backprop Neural Network (MLP)",
            "hidden_layers": str(hidden_layer_sizes),
            "iterations": mlp.n_iter_,
            "final_loss": round(float(mlp.loss_), 4),
            "train_time_sec": round(train_time, 2),
            "split": "validation",
            "accuracy": round(val_acc, 4),
            "macro_f1": round(val_macro, 4),
            "weighted_f1": round(val_weighted, 4),
            "macro_precision": round(precision_score(y_val, val_pred, average="macro", zero_division=0), 4),
            "macro_recall": round(recall_score(y_val, val_pred, average="macro", zero_division=0), 4),
        },
        {
            "model": "Backprop Neural Network (MLP)",
            "hidden_layers": str(hidden_layer_sizes),
            "iterations": mlp.n_iter_,
            "final_loss": round(float(mlp.loss_), 4),
            "train_time_sec": round(train_time, 2),
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

    model_path = config.MODELS_DIR / "bpnn_mlp.joblib"
    joblib.dump(mlp, model_path)
    print(f"\n[MODEL SAVED] {model_path}")

    csv_path = config.RESULTS_DIR / "bpnn_mlp_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"[RESULTS SAVED] {csv_path}")

    # 7. Generate Diagnostic Plots (Loss Curve & Confusion Matrix)
    plot_bpnn_diagnostics(mlp, y_val, val_pred, classes)

    return {
        "model": mlp,
        "metrics_df": df_metrics,
        "val_pred": val_pred,
        "test_pred": test_pred,
    }


def plot_bpnn_diagnostics(
    mlp: MLPClassifier,
    y_val: np.ndarray,
    val_pred: np.ndarray,
    classes: list,
):
    """Generates dual-panel plot of training loss curve and validation confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Loss Curve across Epochs
    axes[0].plot(mlp.loss_curve_, color="#1f77b4", lw=2.2, label="Cross-Entropy Training Loss")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        axes[0].plot(mlp.validation_scores_, color="#ff7f0e", lw=2, linestyle="--", label="Validation Score")
    axes[0].set_title(f"Backpropagation Training Loss Curve ({mlp.n_iter_} Epochs)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch / Iteration", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Right: Normalized Confusion Matrix
    cm_norm = confusion_matrix(y_val, val_pred, labels=classes, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
    disp.plot(cmap="Blues", ax=axes[1], values_format=".2%", colorbar=True)
    axes[1].set_title("Validation Normalized Confusion Matrix", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontsize=10)
    axes[1].set_ylabel("True Class", fontsize=10)

    plt.tight_layout()
    plot_path = config.PLOTS_DIR / "bpnn_loss_and_cm.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT SAVED] {plot_path}")


def main():
    print("[INIT] Loading preprocessed candidate-grouped features...")
    train_feats, val_feats, test_feats = feature_engineering.load_or_create_features()
    train_bpnn_model(train_feats, val_feats, test_feats)
    print("\n[DONE] Experiment 5: Backpropagation Neural Network completed successfully.")


if __name__ == "__main__":
    main()
