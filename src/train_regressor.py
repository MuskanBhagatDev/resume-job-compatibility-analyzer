"""
Regression Model Training and Evaluation Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project - Phase 2 (Step 2 - Corrected).

Trains and benchmarks Linear Regression models to estimate continuous final_score:
1. Linear Regression (Ordinary Least Squares on legitimate structured features)
2. Ridge Regression (L2 Regularization with varying alpha: 1.0, 100.0, 1000.0)

Demonstrates L2 shrinkage dynamics (norm of beta) to explain sample-size scaling:
When N = 85,392 and P = 16, alpha = 1.0 introduces a shrinkage penalty of only
1 / 85,392 (~0.001%), explaining why OLS and Ridge(alpha=1.0) are numerically identical.
Larger alpha (e.g. 1000.0) demonstrates visible regularized shrinkage.

Generates actual-vs-predicted diagnostic plot: outputs/plots/regression_actual_vs_predicted.png.
Exports regression results to outputs/results/regression_model_comparison.csv.
Persists trained regression models to models/ directory.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure path resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """Computes MAE, MSE, RMSE, and R2 regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MAE": round(float(mae), 6),
        "MSE": round(float(mse), 6),
        "RMSE": round(float(rmse), 6),
        "R2_Score": round(float(r2), 6)
    }


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None
) -> Path:
    """
    Generates actual vs predicted diagnostic plot with 45-degree reference line
    and error residual distribution.
    """
    output_path = save_path or (config.PLOTS_DIR / "regression_actual_vs_predicted.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Subsample for clear rendering
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(y_true), size=min(4000, len(y_true)), replace=False)
    y_t_sub = y_true[sub_idx]
    y_p_sub = y_pred[sub_idx]
    
    ax1.scatter(y_t_sub, y_p_sub, alpha=0.35, color="#2b5c8f", edgecolors="none", s=25, label="Test Pairs")
    
    # Ideal 45-degree line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color="#d9381e", linestyle="--", linewidth=2, label="Ideal (y = x)")
    
    # Metric annotations
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    annotation_text = f"R^2 = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}"
    ax1.text(0.05, 0.92, annotation_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#1a4066", alpha=0.9))
             
    ax1.set_title(f"{model_name}: Actual vs. Predicted Compatibility Score", pad=12, fontweight="bold")
    ax1.set_xlabel("Actual final_score", fontweight="bold")
    ax1.set_ylabel("Predicted final_score", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Residual Distribution Plot
    residuals = y_pred - y_true
    ax2.hist(residuals, bins=40, color="#1e7b5e", edgecolor="black", alpha=0.8, density=True)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax2.axvline(residuals.mean(), color="black", linestyle=":", linewidth=1.5, label=f"Mean: {residuals.mean():.4f}")
    
    ax2.set_title("Prediction Error Residuals Distribution", pad=12, fontweight="bold")
    ax2.set_xlabel("Residual Error (Predicted - Actual)", fontweight="bold")
    ax2.set_ylabel("Density", fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT SAVED] {output_path.name}")
    return output_path


def train_and_evaluate_regressors(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    test_data: Dict[str, Any],
    save_models: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Trains OLS and Ridge models with multiple alpha values to evaluate regularization dynamics.
    Evaluates on both Validation set (for model selection) and Test set.
    """
    X_tr = train_data["X_struct"]
    y_tr = train_data["y_reg"]
    
    X_val = val_data["X_struct"]
    y_val = val_data["y_reg"]
    
    X_te = test_data["X_struct"]
    y_te = test_data["y_reg"]
    
    regressors = {
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "Ridge Regression (alpha=100.0)": Ridge(alpha=100.0, random_state=42),
        "Ridge Regression (alpha=1000.0)": Ridge(alpha=1000.0, random_state=42)
    }
    
    trained_models = {}
    test_predictions = {}
    results = []
    
    print("\n" + "=" * 80)
    print("        TRAINING REGRESSION MODELS & REGULARIZATION SHRINKAGE AUDIT      ")
    print("=" * 80)
    
    for name, model in regressors.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        
        coef_norm = float(np.linalg.norm(model.coef_))
        
        # Validation Evaluation
        val_pred = model.predict(X_val)
        val_m = evaluate_regression(y_val, val_pred, name)
        
        # Test Evaluation
        test_pred = model.predict(X_te)
        test_m = evaluate_regression(y_te, test_pred, name)
        
        print(f"\n[REGRESSOR] {name}")
        print(f"  -> Coefficient L2 Norm: {coef_norm:.6f} | Fit Time: {fit_time:.2f}s")
        print(f"  -> Validation: MAE = {val_m['MAE']:.6f} | RMSE = {val_m['RMSE']:.6f} | R^2 = {val_m['R2_Score']:.6f}")
        print(f"  -> Test:       MAE = {test_m['MAE']:.6f} | RMSE = {test_m['RMSE']:.6f} | R^2 = {test_m['R2_Score']:.6f}")
        
        results.append({
            "Model": name,
            "L2_Coef_Norm": round(coef_norm, 6),
            "Val_MAE": val_m["MAE"],
            "Val_RMSE": val_m["RMSE"],
            "Val_R2": val_m["R2_Score"],
            "Test_MAE": test_m["MAE"],
            "Test_RMSE": test_m["RMSE"],
            "Test_R2": test_m["R2_Score"],
            "Fit_Time_Sec": round(fit_time, 2)
        })
        
        trained_models[name] = model
        test_predictions[name] = test_pred
        
        if save_models:
            clean_name = name.lower().split()[0]
            if "1000" in name:
                clean_name = "ridge_alpha1000"
            elif "100" in name:
                clean_name = "ridge_alpha100"
            elif "1.0" in name:
                clean_name = "ridge_alpha1"
            else:
                clean_name = "linear_ols"
            model_path = config.MODELS_DIR / f"{clean_name}_regressor.joblib"
            joblib.dump(model, model_path)
            
    results_df = pd.DataFrame(results)
    
    # Save comparison CSV
    summary_csv = config.RESULTS_DIR / "regression_model_comparison.csv"
    results_df.to_csv(summary_csv, index=False)
    print(f"\n[OUTPUT SAVED] Regression comparison saved to: {summary_csv}")
    
    # Plot diagnostic for best regressor
    best_reg_name = results_df.sort_values(by="Val_RMSE").iloc[0]["Model"]
    plot_actual_vs_predicted(y_te, test_predictions[best_reg_name], best_reg_name)
    
    return trained_models, results_df, test_predictions
