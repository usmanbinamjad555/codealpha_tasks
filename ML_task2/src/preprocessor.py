"""
src/preprocessor.py
===================
Builds a robust, sklearn-compatible preprocessing pipeline that handles:
  - Median imputation for missing numeric values
  - Standard scaling (zero mean, unit variance)
  - Train/validation/test splitting with stratification

The pipeline is fully serialisable with joblib, so fitted transformers can
be saved alongside trained models for consistent inference in production.

Author : CodeAlpha Internship Project
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_generator import TARGET_COLUMN


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TEST_SIZE       : float = 0.20   # 20 % held out for final evaluation
VALIDATION_SIZE : float = 0.15   # 15 % of training set used for validation
RANDOM_SEED     : int   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline() -> Pipeline:
    """
    Construct the numeric feature-preprocessing pipeline.

    Steps
    -----
    1. **SimpleImputer** – fills missing values with the column median,
       which is more robust to outliers than the mean.
    2. **StandardScaler** – standardises features to zero mean / unit
       variance so that distance-based and gradient-based algorithms
       converge faster and treat all features equally.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An *unfitted* pipeline ready for ``fit_transform`` / ``transform``.

    Example
    -------
    >>> pipe = build_preprocessing_pipeline()
    >>> pipe
    Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())])
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]
    )


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate a DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.
    target_col : str
        Name of the target (label) column.

    Returns
    -------
    X : pd.DataFrame
        All columns except *target_col*.
    y : pd.Series
        The target column values.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """
    Stratified three-way split: train / validation / test.

    Stratification preserves the approval-rate ratio across every split,
    which is critical when the target is imbalanced.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    test_size : float
        Fraction of the full dataset reserved for the final test set.
    val_size : float
        Fraction of the *remaining* (post-test) data used for validation.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series

    Example
    -------
    >>> X_tr, X_v, X_te, y_tr, y_v, y_te = split_dataset(X, y)
    >>> len(X_tr) + len(X_v) + len(X_te) == len(X)
    True
    """
    # First split: carve out the held-out test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    # Second split: carve validation from the remaining training pool
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_seed,
    )

    print(
        f"[Preprocessor] Split sizes — "
        f"Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | "
        f"Test: {len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_transform_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the preprocessing pipeline on training data and transform all splits.

    ⚠ The pipeline is **only fitted on X_train** to prevent data leakage.
    Validation and test sets are transformed using training-set statistics.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted preprocessing pipeline.
    X_train, X_val, X_test : pd.DataFrame
        Feature splits.

    Returns
    -------
    X_train_t, X_val_t, X_test_t : np.ndarray
        Transformed numeric arrays ready for model training.
    """
    X_train_t = pipeline.fit_transform(X_train)
    X_val_t   = pipeline.transform(X_val)
    X_test_t  = pipeline.transform(X_test)

    print(
        f"[Preprocessor] Pipeline fitted | "
        f"Output shape: {X_train_t.shape[1]} features"
    )
    return X_train_t, X_val_t, X_test_t
