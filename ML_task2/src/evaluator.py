"""
src/evaluator.py
================
Computes, formats, and saves all evaluation artefacts for trained credit
scoring models:

  Metrics   → Precision, Recall, F1-Score, ROC-AUC, Average Precision
  Reports   → Per-model and side-by-side comparison tables saved to CSV
  Console   → Colour-coded, aligned summary printed to stdout

All functions are stateless — they accept a fitted estimator and data,
compute metrics, and return / print results.  Nothing is mutated.

Author : CodeAlpha Internship Project
License: MIT
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.trainer import TrainingResult


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    result:        TrainingResult,
    X_test:        np.ndarray,
    y_test:        np.ndarray | pd.Series,
    threshold:     float = 0.50,
    report_dir:    str   = "reports",
) -> dict[str, float]:
    """
    Evaluate one trained model on the held-out test set.

    Metrics computed
    ----------------
    * **ROC-AUC** – area under the receiver-operating curve; threshold-free
      measure of ranking quality.  The primary metric for credit models.
    * **Average Precision** – area under the precision-recall curve; more
      informative than ROC-AUC when classes are imbalanced.
    * **Precision** – of all predicted positives, what fraction are truly
      creditworthy?  Minimises unnecessary rejections.
    * **Recall** – of all truly creditworthy applicants, what fraction did we
      approve?  Minimises missed business opportunities.
    * **F1-Score** – harmonic mean of precision and recall.
    * **Accuracy** – overall correct predictions (can be misleading with
      class imbalance; reported for completeness).

    Parameters
    ----------
    result : TrainingResult
        Output of ``ModelTrainer.train_logistic_regression`` or
        ``ModelTrainer.train_random_forest``.
    X_test : np.ndarray
        Pre-processed held-out feature matrix.
    y_test : array-like
        True binary labels for the test set.
    threshold : float
        Decision threshold applied to ``predict_proba`` scores (default 0.5).
        Lower values increase recall; higher values increase precision.
    report_dir : str
        Directory where per-model CSV reports are saved.

    Returns
    -------
    dict[str, float]
        Metric name → value mapping, suitable for comparison tables.
    """
    model     = result.best_estimator
    y_prob    = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_prob >= threshold).astype(int)
    y_test_np = np.asarray(y_test)

    metrics = {
        "Model":              result.model_name,
        "ROC-AUC":            round(roc_auc_score(y_test_np, y_prob),            4),
        "Avg Precision":      round(average_precision_score(y_test_np, y_prob),  4),
        "Precision":          round(precision_score(y_test_np, y_pred,
                                                    zero_division=0),             4),
        "Recall":             round(recall_score(y_test_np, y_pred,
                                                 zero_division=0),               4),
        "F1-Score":           round(f1_score(y_test_np, y_pred,
                                             zero_division=0),                   4),
        "Accuracy":           round(float((y_test_np == y_pred).mean()),         4),
        "CV AUC Mean":        round(result.cv_auc_mean,                          4),
        "CV AUC Std":         round(result.cv_auc_std,                           4),
        "Train Time (s)":     round(result.training_time_s,                      2),
        "Threshold":          threshold,
    }

    _print_model_report(metrics, y_test_np, y_pred, result)
    _save_model_report(metrics, result, report_dir)
    return metrics


def compare_models(
    metrics_list: list[dict[str, float]],
    report_dir:   str = "reports",
) -> pd.DataFrame:
    """
    Build and display a side-by-side comparison table for all evaluated models.

    Parameters
    ----------
    metrics_list : list[dict]
        List of metric dicts returned by ``evaluate_model``.
    report_dir : str
        Directory where the comparison CSV is saved.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by ROC-AUC descending.
    """
    df = pd.DataFrame(metrics_list).set_index("Model")
    df = df.sort_values("ROC-AUC", ascending=False)

    border = "═" * 72
    print(f"\n{border}")
    print("  MODEL COMPARISON — HELD-OUT TEST SET")
    print(border)
    print(df.to_string())
    print(f"{border}\n")

    # Highlight winner
    winner = df["ROC-AUC"].idxmax()
    print(f"  🏆  Best model by ROC-AUC: {winner}  ({df.loc[winner, 'ROC-AUC']:.4f})")
    print(f"{border}\n")

    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "model_comparison.csv")
    df.to_csv(out_path)
    print(f"[Evaluator] Comparison saved → {out_path}")
    return df


def print_feature_importances(result: TrainingResult, top_n: int = 10) -> None:
    """
    Print the top *top_n* most important features for a trained model.

    Parameters
    ----------
    result : TrainingResult
        Trained model result containing feature importances.
    top_n : int
        Number of top features to display. Default: 10.
    """
    if result.feature_importances is None:
        print(f"[Evaluator] No feature importances available for {result.model_name}.")
        return

    top = result.feature_importances.head(top_n)
    border = "─" * 50

    print(f"\n  Feature Importances — {result.model_name} (top {top_n})")
    print(f"  {border}")
    max_score = top.max() or 1.0

    for feat, score in top.items():
        bar_len = int((score / max_score) * 30)
        bar     = "█" * bar_len
        print(f"  {feat:<40} {score:>8.4f}  {bar}")
    print(f"  {border}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_model_report(
    metrics:  dict[str, float],
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    result:   TrainingResult,
) -> None:
    """Print a formatted per-model evaluation to stdout."""
    border  = "─" * 60
    h_border = "═" * 60

    print(f"\n{h_border}")
    print(f"  EVALUATION REPORT — {metrics['Model'].upper()}")
    print(h_border)

    # Core metrics table
    metric_rows = [
        ("ROC-AUC",       metrics["ROC-AUC"]),
        ("Avg Precision",  metrics["Avg Precision"]),
        ("Precision",      metrics["Precision"]),
        ("Recall",         metrics["Recall"]),
        ("F1-Score",       metrics["F1-Score"]),
        ("Accuracy",       metrics["Accuracy"]),
    ]
    print(f"\n  {'Metric':<25} {'Test Set':>10}   {'CV Mean':>10}  {'CV Std':>8}")
    print(f"  {border}")
    print(f"  {'ROC-AUC':<25} {metrics['ROC-AUC']:>10.4f}   "
          f"{metrics['CV AUC Mean']:>10.4f}  {metrics['CV AUC Std']:>8.4f}")
    for name, val in metric_rows[1:]:
        print(f"  {name:<25} {val:>10.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix (threshold = {metrics['Threshold']:.2f})")
    print(f"  {border}")
    print(f"  {'':20}  Pred: Rejected   Pred: Approved")
    print(f"  {'True: Rejected':20}  {cm[0,0]:>14,}   {cm[0,1]:>14,}")
    print(f"  {'True: Approved':20}  {cm[1,0]:>14,}   {cm[1,1]:>14,}")

    # Sklearn classification report
    print(f"\n  Per-Class Report")
    print(f"  {border}")
    cr = classification_report(
        y_true, y_pred,
        target_names=["Rejected (0)", "Approved (1)"],
        zero_division=0,
    )
    for line in cr.split("\n"):
        print(f"  {line}")

    print(f"\n  Best Hyperparameters: {result.best_params}")
    print(f"  Training Time       : {metrics['Train Time (s)']:.2f}s")
    print(h_border)


def _save_model_report(
    metrics:    dict[str, float],
    result:     TrainingResult,
    report_dir: str,
) -> None:
    """Save a single-row metric CSV for the given model."""
    os.makedirs(report_dir, exist_ok=True)
    safe_name = result.model_name.lower().replace(" ", "_")
    out_path  = os.path.join(report_dir, f"{safe_name}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    print(f"[Evaluator] Report saved → {out_path}")

    # Save feature importances if available
    if result.feature_importances is not None:
        fi_path = os.path.join(report_dir, f"{safe_name}_feature_importances.csv")
        result.feature_importances.to_csv(fi_path, header=True)
        print(f"[Evaluator] Feature importances saved → {fi_path}")
