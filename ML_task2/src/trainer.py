"""
src/trainer.py
==============
Trains and cross-validates two complementary credit-scoring models:

  1. **Logistic Regression** – a fast, interpretable linear baseline that
     returns well-calibrated probabilities and is easy to audit.
  2. **Random Forest** – an ensemble of decision trees that captures
     non-linear interactions and feature importances without needing
     manual feature selection.

Both models are wrapped in a thin ``ModelTrainer`` class that provides a
consistent ``.fit()`` / ``.predict()`` / ``.predict_proba()`` interface,
hyperparameter tuning via ``GridSearchCV``, and cross-validated AUC scoring.

Author : CodeAlpha Internship Project
License: MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


# ─────────────────────────────────────────────────────────────────────────────
# Default hyper-parameter grids
# ─────────────────────────────────────────────────────────────────────────────

LR_PARAM_GRID: dict[str, list] = {
    "C":          [0.01, 0.1, 1.0, 10.0],
    # sklearn ≥ 1.8: use l1_ratio (0 = L2, 1 = L1) with elasticnet solver
    # For simplicity and full compatibility we keep only C and solver tuning;
    # regularisation type is fixed to L2 (the most common choice in credit).
    "solver":     ["lbfgs", "liblinear"],
}

RF_PARAM_GRID: dict[str, list] = {
    "n_estimators":      [100, 200],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "max_features":      ["sqrt", "log2"],
}

CV_FOLDS: int = 5
RANDOM_SEED: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data container for training results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """
    Holds every artefact produced during a single model-training run.

    Attributes
    ----------
    model_name : str
        Human-readable identifier (e.g. ``"Logistic Regression"``).
    best_estimator : Any
        The best fitted estimator found by ``GridSearchCV``.
    best_params : dict
        Hyperparameters of the best estimator.
    cv_auc_scores : np.ndarray
        Cross-validation ROC-AUC scores (one per fold).
    cv_auc_mean : float
        Mean cross-validation ROC-AUC.
    cv_auc_std : float
        Standard deviation of cross-validation ROC-AUC scores.
    training_time_s : float
        Wall-clock training time in seconds.
    feature_importances : pd.Series or None
        Feature importance / coefficient magnitudes (None for unsupported models).
    """
    model_name:          str
    best_estimator:      Any
    best_params:         dict
    cv_auc_scores:       np.ndarray
    cv_auc_mean:         float
    cv_auc_std:          float
    training_time_s:     float
    feature_importances: pd.Series | None = field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Core trainer class
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Unified interface for fitting, tuning, and cross-validating credit models.

    Parameters
    ----------
    random_seed : int
        Seed for all stochastic components. Default: 42.
    cv_folds : int
        Number of stratified cross-validation folds. Default: 5.
    scoring : str
        Scikit-learn scoring metric used during grid search. Default: ``"roc_auc"``.
    n_jobs : int
        Number of parallel workers for grid search. Default: -1 (all CPUs).

    Example
    -------
    >>> trainer = ModelTrainer()
    >>> lr_result = trainer.train_logistic_regression(X_train, y_train, feature_names)
    >>> rf_result = trainer.train_random_forest(X_train, y_train, feature_names)
    """

    def __init__(
        self,
        random_seed: int = RANDOM_SEED,
        cv_folds:    int = CV_FOLDS,
        scoring:     str = "roc_auc",
        n_jobs:      int = -1,
    ) -> None:
        self.random_seed = random_seed
        self.cv_folds    = cv_folds
        self.scoring     = scoring
        self.n_jobs      = n_jobs
        self._cv_splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_seed
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def train_logistic_regression(
        self,
        X_train:       np.ndarray,
        y_train:       np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
        param_grid:    dict | None = None,
    ) -> TrainingResult:
        """
        Fit a Logistic Regression with L1/L2 regularisation via grid search.

        Logistic Regression is the industry-standard *interpretable* baseline
        for credit scoring.  Its coefficients directly encode feature impact on
        log-odds of approval, satisfying explainability requirements of
        regulators such as the CFPB.

        Parameters
        ----------
        X_train : np.ndarray
            Pre-processed training feature matrix.
        y_train : array-like
            Binary training labels (0 = rejected, 1 = approved).
        feature_names : list[str] or None
            Column names for reporting feature importances.
        param_grid : dict or None
            Custom hyperparameter grid.  Defaults to ``LR_PARAM_GRID``.

        Returns
        -------
        TrainingResult
            Full training artefacts including the best estimator.
        """
        base_model = LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",   # handles class imbalance automatically
            random_state=self.random_seed,
        )
        grid = param_grid or LR_PARAM_GRID
        return self._fit_with_grid_search(
            base_model, grid, X_train, y_train,
            model_name="Logistic Regression",
            feature_names=feature_names,
        )

    def train_random_forest(
        self,
        X_train:       np.ndarray,
        y_train:       np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
        param_grid:    dict | None = None,
    ) -> TrainingResult:
        """
        Fit a Random Forest ensemble via grid search.

        Random Forest captures non-linear feature interactions and naturally
        handles heterogeneous feature scales.  Its built-in feature importances
        (mean impurity decrease) provide an additional transparency layer.

        Parameters
        ----------
        X_train : np.ndarray
            Pre-processed training feature matrix.
        y_train : array-like
            Binary training labels.
        feature_names : list[str] or None
            Column names for reporting feature importances.
        param_grid : dict or None
            Custom hyperparameter grid.  Defaults to ``RF_PARAM_GRID``.

        Returns
        -------
        TrainingResult
            Full training artefacts including the best estimator.
        """
        base_model = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
        )
        grid = param_grid or RF_PARAM_GRID
        return self._fit_with_grid_search(
            base_model, grid, X_train, y_train,
            model_name="Random Forest",
            feature_names=feature_names,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fit_with_grid_search(
        self,
        base_model,
        param_grid:    dict,
        X_train:       np.ndarray,
        y_train:       np.ndarray | pd.Series,
        model_name:    str,
        feature_names: list[str] | None,
    ) -> TrainingResult:
        """
        Run ``GridSearchCV`` then compute out-of-fold cross-validation AUC on
        the best estimator.

        Internal workflow
        -----------------
        1. ``GridSearchCV`` searches *param_grid* using stratified k-fold CV,
           selecting the configuration with the highest mean ROC-AUC.
        2. The best estimator is re-evaluated with ``cross_val_score`` so we
           record per-fold AUC variance separately from the grid-search result.
        3. Feature importances are extracted where available.
        """
        print(f"\n[Trainer] Fitting {model_name} …")
        t_start = time.perf_counter()

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self._cv_splitter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Cross-validate best estimator independently for unbiased AUC estimates
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=self._cv_splitter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )

        elapsed = time.perf_counter() - t_start

        # Feature importance extraction
        importances = self._extract_importances(best_model, feature_names)

        result = TrainingResult(
            model_name=model_name,
            best_estimator=best_model,
            best_params=grid_search.best_params_,
            cv_auc_scores=cv_scores,
            cv_auc_mean=float(cv_scores.mean()),
            cv_auc_std=float(cv_scores.std()),
            training_time_s=elapsed,
            feature_importances=importances,
        )

        print(
            f"[Trainer] {model_name} done in {elapsed:.1f}s | "
            f"CV AUC: {result.cv_auc_mean:.4f} ± {result.cv_auc_std:.4f} | "
            f"Best params: {result.best_params}"
        )
        return result

    @staticmethod
    def _extract_importances(
        model,
        feature_names: list[str] | None,
    ) -> pd.Series | None:
        """
        Extract feature importances or coefficient magnitudes.

        - **LogisticRegression**: absolute values of the coefficients.
        - **RandomForestClassifier**: mean impurity decrease (Gini importance).
        - Other models: returns ``None``.
        """
        if feature_names is None:
            return None

        if hasattr(model, "feature_importances_"):
            scores = model.feature_importances_
        elif hasattr(model, "coef_"):
            scores = np.abs(model.coef_[0])
        else:
            return None

        return (
            pd.Series(scores, index=feature_names)
            .rename("importance")
            .sort_values(ascending=False)
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.data_generator import generate_credit_dataset, TARGET_COLUMN
    from src.feature_engineer import engineer_features
    from src.preprocessor import (
        build_preprocessing_pipeline,
        fit_transform_pipeline,
        split_dataset,
        split_features_target,
    )

    raw      = generate_credit_dataset(n_samples=2_000)
    X_raw, y = split_features_target(raw)
    X_eng    = engineer_features(X_raw)

    X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(X_eng, y)
    pipe = build_preprocessing_pipeline()
    X_tr_t, X_v_t, X_te_t = fit_transform_pipeline(pipe, X_tr, X_v, X_te)

    trainer   = ModelTrainer()
    feat_names = list(X_eng.columns)

    lr_result = trainer.train_logistic_regression(X_tr_t, y_tr, feat_names)
    rf_result = trainer.train_random_forest(X_tr_t, y_tr, feat_names)

    print("\nTop-5 LR features:\n", lr_result.feature_importances.head())
    print("\nTop-5 RF features:\n", rf_result.feature_importances.head())
