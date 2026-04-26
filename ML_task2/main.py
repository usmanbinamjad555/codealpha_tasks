"""
main.py
=======
End-to-end Credit Scoring Model pipeline orchestrator.

Execution order
---------------
1.  Generate synthetic dataset          → data_generator.py
2.  Engineer domain features            → feature_engineer.py
3.  Split into train / val / test       → preprocessor.py
4.  Fit preprocessing pipeline          → preprocessor.py
5.  Train Logistic Regression           → trainer.py
6.  Train Random Forest                 → trainer.py
7.  Evaluate both models on test set    → evaluator.py
8.  Print side-by-side comparison       → evaluator.py
9.  Save models to disk                 → joblib

Run
---
    python main.py                   # default 5 000 samples
    python main.py --samples 10000   # custom sample count
    python main.py --no-save         # skip saving models

Author : CodeAlpha Internship Project
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np

# ── Local imports ─────────────────────────────────────────────────────────────
from src.data_generator  import generate_credit_dataset, TARGET_COLUMN
from src.feature_engineer import engineer_features
from src.preprocessor    import (
    build_preprocessing_pipeline,
    fit_transform_pipeline,
    split_dataset,
    split_features_target,
)
from src.trainer    import ModelTrainer
from src.evaluator  import compare_models, evaluate_model, print_feature_importances


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = "data"
MODEL_DIR   = "models"
REPORT_DIR  = "reports"
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    n_samples:  int  = 5_000,
    save_models: bool = True,
) -> dict:
    """
    Execute the full credit-scoring pipeline and return all artefacts.

    Parameters
    ----------
    n_samples : int
        Number of synthetic applicant records to generate.
    save_models : bool
        Whether to persist fitted models and preprocessing pipeline to disk.

    Returns
    -------
    dict
        Keys: ``"lr_result"``, ``"rf_result"``, ``"comparison"``,
              ``"pipeline"``, ``"feature_names"``.
    """
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    _print_banner()

    # ── Step 1 · Data generation ──────────────────────────────────────────────
    _section("1 / 6  Data Generation")
    raw_df = generate_credit_dataset(
        n_samples=n_samples,
        random_seed=RANDOM_SEED,
        output_path=os.path.join(DATA_DIR, "raw_credit_data.csv"),
    )

    # ── Step 2 · Feature engineering ─────────────────────────────────────────
    _section("2 / 6  Feature Engineering")
    X_raw, y  = split_features_target(raw_df, TARGET_COLUMN)
    X_enriched = engineer_features(X_raw)
    feature_names = list(X_enriched.columns)
    print(f"[Pipeline] Feature matrix shape: {X_enriched.shape}")

    # ── Step 3 · Train / Val / Test split ─────────────────────────────────────
    _section("3 / 6  Dataset Splitting")
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(
        X_enriched, y, random_seed=RANDOM_SEED
    )
    _print_class_balance("Train", y_tr)
    _print_class_balance("Test",  y_te)

    # ── Step 4 · Preprocessing pipeline ──────────────────────────────────────
    _section("4 / 6  Preprocessing (Impute → Scale)")
    preproc_pipeline = build_preprocessing_pipeline()
    X_tr_t, X_v_t, X_te_t = fit_transform_pipeline(
        preproc_pipeline, X_tr, X_v, X_te
    )

    # ── Step 5 · Model training ───────────────────────────────────────────────
    _section("5 / 6  Model Training")
    trainer = ModelTrainer(random_seed=RANDOM_SEED)

    lr_result = trainer.train_logistic_regression(
        X_tr_t, y_tr, feature_names=feature_names
    )
    rf_result = trainer.train_random_forest(
        X_tr_t, y_tr, feature_names=feature_names
    )

    # ── Step 6 · Evaluation ───────────────────────────────────────────────────
    _section("6 / 6  Evaluation on Held-out Test Set")
    lr_metrics = evaluate_model(lr_result, X_te_t, y_te, report_dir=REPORT_DIR)
    rf_metrics = evaluate_model(rf_result, X_te_t, y_te, report_dir=REPORT_DIR)

    comparison = compare_models([lr_metrics, rf_metrics], report_dir=REPORT_DIR)

    print_feature_importances(lr_result, top_n=10)
    print_feature_importances(rf_result, top_n=10)

    # ── Validation set quick-check ────────────────────────────────────────────
    _validation_check(lr_result, rf_result, X_v_t, y_v)

    # ── Persist artefacts ─────────────────────────────────────────────────────
    if save_models:
        _save_artefacts(preproc_pipeline, lr_result, rf_result)

    print("\n✅  Pipeline complete.  Check the reports/ directory for CSV outputs.\n")

    return {
        "lr_result":     lr_result,
        "rf_result":     rf_result,
        "comparison":    comparison,
        "pipeline":      preproc_pipeline,
        "feature_names": feature_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          CodeAlpha — Credit Scoring Model Pipeline               ║
║          Libraries: pandas · numpy · scikit-learn                ║
╚══════════════════════════════════════════════════════════════════╝
""")


def _section(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


def _print_class_balance(split_name: str, y: object) -> None:
    import pandas as pd
    s = pd.Series(y).value_counts(normalize=True)
    print(
        f"[Pipeline] {split_name} class balance — "
        f"Rejected: {s.get(0, 0):.1%} | Approved: {s.get(1, 0):.1%}"
    )


def _validation_check(lr_result, rf_result, X_val_t, y_val) -> None:
    """Quick val-set AUC check to confirm no over-fitting."""
    from sklearn.metrics import roc_auc_score
    import numpy as np

    y_val_np = np.asarray(y_val)
    lr_auc = roc_auc_score(y_val_np, lr_result.best_estimator.predict_proba(X_val_t)[:, 1])
    rf_auc = roc_auc_score(y_val_np, rf_result.best_estimator.predict_proba(X_val_t)[:, 1])

    print(f"\n[Pipeline] Validation AUC — LR: {lr_auc:.4f} | RF: {rf_auc:.4f}")
    gap_lr = abs(lr_result.cv_auc_mean - lr_auc)
    gap_rf = abs(rf_result.cv_auc_mean - rf_auc)
    if gap_lr > 0.05:
        print(f"  ⚠  Logistic Regression CV↔Val gap = {gap_lr:.3f} — possible overfit!")
    if gap_rf > 0.05:
        print(f"  ⚠  Random Forest CV↔Val gap = {gap_rf:.3f} — possible overfit!")


def _save_artefacts(pipeline, lr_result, rf_result) -> None:
    """Persist the preprocessing pipeline and both fitted models."""
    jobs = [
        (pipeline,                  os.path.join(MODEL_DIR, "preprocessing_pipeline.joblib")),
        (lr_result.best_estimator,  os.path.join(MODEL_DIR, "logistic_regression.joblib")),
        (rf_result.best_estimator,  os.path.join(MODEL_DIR, "random_forest.joblib")),
    ]
    for obj, path in jobs:
        joblib.dump(obj, path)
        print(f"[Pipeline] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeAlpha Credit Scoring Model — end-to-end ML pipeline"
    )
    parser.add_argument(
        "--samples", type=int, default=5_000,
        help="Number of synthetic applicant records to generate (default: 5000)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip persisting trained models to disk"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        n_samples=args.samples,
        save_models=not args.no_save,
    )
