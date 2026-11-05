"""
Classifier Training and Evaluation Module for Resume Job Compatibility Analyzer.
Academic ML Lab Mini-Project - Phase 2 (Step 2 - Corrected).

Implements:
1. Experiment A: Classifiers using All Features (1,516 dimensions: TF-IDF + Structured)
2. Experiment B: Classifiers using SelectKBest Features (500 dimensions)
3. Specialized Models: KNN and Random Forest on Structured Features, Naive Bayes on TF-IDF
4. Strict Academic Validation Protocol:
   - Model selection performed strictly on Validation Data
   - Test data evaluated for final unbiased benchmark
5. Persists all trained models to models/ directory
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# Ensure path resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def get_all_classifier_experiments(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns full catalog of classifier configurations covering both
    all-feature and selectKBest feature selection experiments.
    """
    return {
        # --- Experiment A: All Features (1,516 features) ---
        "Logistic Regression (All Features)": {
            "model": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                solver="lbfgs"
            ),
            "feature_set": "X_all",
            "feature_type": "All Features (1,516)",
            "description": "L2 Logistic Regression on combined TF-IDF + structured features."
        },
        "Support Vector Machine (All Features)": {
            "model": LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                max_iter=2000,
                dual="auto"
            ),
            "feature_set": "X_all",
            "feature_type": "All Features (1,516)",
            "description": "Linear Support Vector Classifier on all 1,516 features."
        },
        
        # --- Experiment B: SelectKBest Features (500 features) ---
        "Logistic Regression (Selected Features)": {
            "model": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                solver="lbfgs"
            ),
            "feature_set": "X_selected",
            "feature_type": "Selected Features (500)",
            "description": "L2 Logistic Regression on top 500 features selected by ANOVA F-score."
        },
        "Support Vector Machine (Selected Features)": {
            "model": LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                max_iter=2000,
                dual="auto"
            ),
            "feature_set": "X_selected",
            "feature_type": "Selected Features (500)",
            "description": "Linear Support Vector Classifier on top 500 features selected by ANOVA F-score."
        },
        
        # --- Specialized Representations ---
        "Naive Bayes (TF-IDF Text)": {
            "model": ComplementNB(alpha=1.0, norm=False),
            "feature_set": "X_tfidf",
            "feature_type": "TF-IDF Text (1,500)",
            "description": "Complement Naive Bayes on non-negative sparse TF-IDF text features."
        },
        "K-Nearest Neighbors (Structured)": {
            "model": KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1
            ),
            "feature_set": "X_struct",
            "feature_type": "Structured (16)",
            "description": "Distance-weighted KNN operating on 16 normalized structured/interaction dimensions."
        },
        "Random Forest (Structured)": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            "feature_set": "X_struct",
            "feature_type": "Structured (16)",
            "description": "100-tree bagging ensemble with balanced subsample weighting."
        }
    }


def evaluate_classifier_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    labels = target_names or ["weak_positive", "medium_positive", "strong_positive"]
    
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro_F1": round(f1_macro, 4),
        "Weighted_F1": round(f1_weighted, 4),
        "Macro_Precision": round(prec_macro, 4),
        "Macro_Recall": round(rec_macro, 4),
        "Weighted_Precision": round(prec_weighted, 4),
        "Weighted_Recall": round(rec_weighted, 4),
        "Confusion_Matrix": cm,
        "Classification_Report": report_dict
    }


def train_and_evaluate_classifiers(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    test_data: Dict[str, Any],
    save_models: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Trains all classifier configurations on training data.
    Evaluates on Validation set for model selection, and on Test set for final benchmark.
    """
    models_def = get_all_classifier_experiments(random_state=42)
    trained_models = {}
    test_predictions = {}
    val_predictions = {}
    results_rows = []
    
    y_train = train_data["y_clf"]
    y_val = val_data["y_clf"]
    y_test = test_data["y_clf"]
    
    target_classes = ["weak_positive", "medium_positive", "strong_positive"]
    
    print("\n" + "=" * 80)
    print("      TRAINING & BENCHMARKING CLASSIFIERS (VALIDATION-FIRST PROTOCOL)   ")
    print("=" * 80)
    
    for name, config_dict in models_def.items():
        feat_key = config_dict["feature_set"]
        feat_type = config_dict["feature_type"]
        clf = config_dict["model"]
        
        X_tr = train_data[feat_key]
        X_val = val_data[feat_key]
        X_te = test_data[feat_key]
        
        print(f"\n[TRAINING] {name}")
        print(f"  -> Representation: {feat_type} | Input Shape: {X_tr.shape}")
        t0 = time.time()
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t0
        
        # 1. Validation Evaluation (Used for Model Selection)
        t_val = time.time()
        val_pred = clf.predict(X_val)
        val_metrics = evaluate_classifier_predictions(y_val, val_pred, name, target_classes)
        val_predictions[name] = val_pred
        
        # 2. Test Evaluation (Unbiased Benchmark)
        t_te = time.time()
        test_pred = clf.predict(X_te)
        test_time = time.time() - t_te
        test_metrics = evaluate_classifier_predictions(y_test, test_pred, name, target_classes)
        test_predictions[name] = test_pred
        
        print(f"  -> Fit Time: {fit_time:.2f}s | Predict Time (Val): {time.time()-t_val:.2f}s")
        print(f"  -> Validation: Accuracy = {val_metrics['Accuracy']:.4f} | Macro F1 = {val_metrics['Macro_F1']:.4f} | Weighted F1 = {val_metrics['Weighted_F1']:.4f}")
        print(f"  -> Test:       Accuracy = {test_metrics['Accuracy']:.4f} | Macro F1 = {test_metrics['Macro_F1']:.4f} | Weighted F1 = {test_metrics['Weighted_F1']:.4f}")
        
        # Per-class F1 on test
        rep_val = val_metrics["Classification_Report"]
        rep_te = test_metrics["Classification_Report"]
        
        results_rows.append({
            "Model": name,
            "Feature_Set": feat_type,
            "Val_Accuracy": val_metrics["Accuracy"],
            "Val_Macro_F1": val_metrics["Macro_F1"],
            "Val_Weighted_F1": val_metrics["Weighted_F1"],
            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Macro_F1": test_metrics["Macro_F1"],
            "Test_Weighted_F1": test_metrics["Weighted_F1"],
            "Test_Macro_Precision": test_metrics["Macro_Precision"],
            "Test_Macro_Recall": test_metrics["Macro_Recall"],
            "Test_F1_Weak": round(rep_te.get("weak_positive", {}).get("f1-score", 0.0), 4),
            "Test_F1_Medium": round(rep_te.get("medium_positive", {}).get("f1-score", 0.0), 4),
            "Test_F1_Strong": round(rep_te.get("strong_positive", {}).get("f1-score", 0.0), 4),
            "Fit_Time_Sec": round(fit_time, 2),
            "Confusion_Matrix": test_metrics["Confusion_Matrix"]
        })
        
        trained_models[name] = clf
        
        if save_models:
            clean_name = name.lower().split("(")[0].strip().replace(" ", "_").replace("-", "_")
            if "selected" in name.lower():
                clean_name += "_selected"
            model_path = config.MODELS_DIR / f"{clean_name}.joblib"
            joblib.dump(clf, model_path)
            
    results_df = pd.DataFrame(results_rows)
    return trained_models, results_df, test_predictions
