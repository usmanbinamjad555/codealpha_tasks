"""
model_trainer.py
================
Trains and optionally tunes two classifiers:
  - RandomForestClassifier  (sklearn)
  - XGBoostClassifier       (xgboost)

Both models are evaluated on the validation set during training.
Hyperparameter search uses RandomizedSearchCV to keep runtimes practical.
Fitted models are serialised with joblib for use by the Predictor.

Design notes
------------
* ModelTrainer takes pre-split, pre-scaled numpy arrays — it has ZERO
  knowledge of raw data or the scaler. This strict separation means the
  Predictor only needs to load the model file; all pre-processing is the
  Predictor's responsibility via the saved scaler.
* The 'best_model_name' attribute is set after compare() so downstream
  code can programmatically select the winner for deployment.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CV_FOLDS = 5


class ModelTrainer:
    """
    Train, tune, and persist Random Forest and XGBoost classifiers.

    Parameters
    ----------
    tune_hyperparams : bool
        If True, runs RandomizedSearchCV for both models (~2 min extra).
        If False, uses well-tested default parameters (fast, still accurate).
    n_iter_search : int
        Number of parameter combinations to try per model during search.
    random_state : int
        Global seed.

    Attributes (after train())
    --------------------------
    models        : dict[str, estimator]  — fitted model objects
    val_scores    : dict[str, float]      — validation ROC-AUC per model
    best_model_name : str                 — name of the top-performing model
    """

    def __init__(
        self,
        tune_hyperparams: bool = True,
        n_iter_search: int = 30,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.tune_hyperparams = tune_hyperparams
        self.n_iter_search = n_iter_search
        self.random_state = random_state

        self.models: Dict[str, object] = {}
        self.val_scores: Dict[str, float] = {}
        self.best_model_name: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "ModelTrainer":
        """
        Train both models on (X_train, y_train) and score on (X_val, y_val).

        Returns self to allow chaining:
            trainer = ModelTrainer().train(X_tr, y_tr, X_v, y_v)
        """
        logger.info("=" * 60)
        logger.info("MODEL TRAINER — starting")
        logger.info("=" * 60)

        self._train_random_forest(X_train, y_train, X_val, y_val)
        self._train_xgboost(X_train, y_train, X_val, y_val)
        self._select_best()

        return self

    def save_model(self, model_name: str, path: str) -> None:
        """Persist a fitted model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], path)
        logger.info("Model '%s' saved -> %s", model_name, path)

    def save_best(self, path: str) -> None:
        """Convenience: save whichever model scored highest on validation."""
        if self.best_model_name is None:
            raise RuntimeError("Call train() before save_best().")
        self.save_model(self.best_model_name, path)
        logger.info("Best model (%s) saved -> %s", self.best_model_name, path)

    # ── Private helpers ────────────────────────────────────────────────────

    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        logger.info("--- Training Random Forest ---")
        t0 = time.time()

        if self.tune_hyperparams:
            param_dist = {
                "n_estimators":      [100, 200, 300, 500],
                "max_depth":         [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
                "max_features":      ["sqrt", "log2", 0.5],
                "class_weight":      [None, "balanced"],
            }
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=self.random_state)
            search = RandomizedSearchCV(
                estimator=RandomForestClassifier(random_state=self.random_state),
                param_distributions=param_dist,
                n_iter=self.n_iter_search,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info("RF best params: %s", search.best_params_)
        else:
            # Sensible defaults — performs well without tuning on this dataset
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

        val_auc = self._roc_auc(model, X_val, y_val)
        elapsed = time.time() - t0

        self.models["random_forest"] = model
        self.val_scores["random_forest"] = val_auc
        logger.info("Random Forest | val ROC-AUC = %.4f | %.1fs", val_auc, elapsed)

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        logger.info("--- Training XGBoost ---")
        t0 = time.time()

        # Compute scale_pos_weight to handle class imbalance
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        spw   = n_neg / max(n_pos, 1)

        if self.tune_hyperparams:
            param_dist = {
                "n_estimators":   [100, 200, 300, 500],
                "max_depth":      [3, 4, 5, 6, 7],
                "learning_rate":  [0.01, 0.05, 0.1, 0.2],
                "subsample":      [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha":      [0, 0.01, 0.1, 1.0],
                "reg_lambda":     [0.1, 1.0, 5.0, 10.0],
                "min_child_weight": [1, 3, 5],
            }
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=self.random_state)
            base = XGBClassifier(
                scale_pos_weight=spw,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=self.random_state,
                verbosity=0,
            )
            search = RandomizedSearchCV(
                estimator=base,
                param_distributions=param_dist,
                n_iter=self.n_iter_search,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info("XGB best params: %s", search.best_params_)
        else:
            model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=self.random_state,
                verbosity=0,
            )
            model.fit(X_train, y_train)

        val_auc = self._roc_auc(model, X_val, y_val)
        elapsed = time.time() - t0

        self.models["xgboost"] = model
        self.val_scores["xgboost"] = val_auc
        logger.info("XGBoost       | val ROC-AUC = %.4f | %.1fs", val_auc, elapsed)

    def _select_best(self) -> None:
        self.best_model_name = max(self.val_scores, key=self.val_scores.get)
        logger.info(
            "Best model on validation set: %s (ROC-AUC=%.4f)",
            self.best_model_name,
            self.val_scores[self.best_model_name],
        )

    @staticmethod
    def _roc_auc(model, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        proba = model.predict_proba(X)[:, 1]
        return roc_auc_score(y, proba)
