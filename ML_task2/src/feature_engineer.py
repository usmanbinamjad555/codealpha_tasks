"""
src/feature_engineer.py
=======================
Transforms the raw feature matrix into an enriched representation by
computing domain-driven interaction terms, ratio features, and risk flags
that are not directly observable in the original columns.

All transformations use only pandas and numpy — no extra dependencies.

Design principles
-----------------
* **Stateless** – every function is a pure transformation; no fitting is
  required, so there is no risk of leakage between splits.
* **Transparent** – every new feature has a clear financial interpretation,
  making the model explainable to credit-committee stakeholders.
* **Idempotent** – calling the function on an already-engineered DataFrame
  will raise a ``ValueError`` rather than silently producing duplicate columns.

Author : CodeAlpha Internship Project
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Engineered feature names (for downstream reference)
# ─────────────────────────────────────────────────────────────────────────────

ENGINEERED_FEATURES: list[str] = [
    # Ratio features
    "loan_to_income_ratio",
    "monthly_debt_to_income",
    "credit_limit_to_income",
    # Interaction / composite features
    "risk_composite_score",
    "credit_experience_score",
    # Binary risk flags
    "flag_high_utilization",
    "flag_multiple_late_payments",
    "flag_excessive_inquiries",
    "flag_high_dti",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce an enriched copy of *df* with additional domain-driven columns.

    All new columns are appended to the existing ones; no original column is
    modified or removed.

    New Features
    ------------
    **Ratio features** — normalise absolute monetary amounts so that a
    $10 000 debt means something different for a $20 000 vs $200 000 earner:

    * ``loan_to_income_ratio`` – requested loan ÷ annual income.
      Higher values indicate the applicant is stretching their budget.
    * ``monthly_debt_to_income`` – monthly debt payments ÷ monthly income.
      Equivalent to the lender-standard front-end DTI.
    * ``credit_limit_to_income`` – total credit limit ÷ annual income.
      Proxy for how much credit an applicant has been trusted with historically.

    **Composite / interaction features** — capture non-linear relationships:

    * ``risk_composite_score`` – weighted combination of the three strongest
      negative predictors (DTI, utilisation, late payments).  Higher = riskier.
    * ``credit_experience_score`` – product of payment history and the log of
      account age in months.  Rewards long *and* clean credit histories.

    **Binary risk flags** — hard thresholds common in credit policy rules:

    * ``flag_high_utilization``        – utilisation ≥ 70 %
    * ``flag_multiple_late_payments``  – ≥ 3 late payments in past 2 years
    * ``flag_excessive_inquiries``     – ≥ 4 hard enquiries in past 6 months
    * ``flag_high_dti``                – DTI ≥ 43 % (Federal QM threshold)

    Parameters
    ----------
    df : pd.DataFrame
        Raw (pre-scaled) feature DataFrame. Must contain the columns produced
        by ``data_generator.generate_credit_dataset``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with all original columns plus the engineered features.

    Raises
    ------
    ValueError
        If the first engineered feature column already exists in *df*,
        indicating the function has already been applied.

    Example
    -------
    >>> from src.data_generator import generate_credit_dataset
    >>> from src.feature_engineer import engineer_features
    >>> raw = generate_credit_dataset(n_samples=500)
    >>> enriched = engineer_features(raw.drop(columns=["credit_approved"]))
    >>> "risk_composite_score" in enriched.columns
    True
    """
    if ENGINEERED_FEATURES[0] in df.columns:
        raise ValueError(
            "Feature engineering has already been applied to this DataFrame. "
            "Pass the original un-engineered DataFrame."
        )

    df = df.copy()

    # ── Ratio features ────────────────────────────────────────────────────────
    eps = 1e-9   # avoid zero-division

    df["loan_to_income_ratio"] = (
        df["loan_amount_requested"] / (df["annual_income"] + eps)
    )

    df["monthly_debt_to_income"] = (
        df["monthly_debt_payments"] / ((df["annual_income"] / 12) + eps)
    )

    df["credit_limit_to_income"] = (
        df["total_credit_limit"] / (df["annual_income"] + eps)
    )

    # ── Composite / interaction features ──────────────────────────────────────
    # risk_composite_score: equally weighted sum of three key risk drivers
    # (each already on a roughly comparable scale after clipping)
    df["risk_composite_score"] = (
        df["debt_to_income_ratio"].clip(0, 2) * 0.40
        + df["credit_utilization_ratio"].clip(0, 1) * 0.35
        + (df["num_late_payments_last_2yr"].clip(0, 14) / 14) * 0.25
    )

    # credit_experience_score: payment quality × log(history length)
    log_age = np.log1p(df["oldest_account_age_months"].clip(lower=0))
    df["credit_experience_score"] = (
        df["payment_history_score"].clip(0, 100) / 100
    ) * log_age

    # ── Binary risk flags ─────────────────────────────────────────────────────
    df["flag_high_utilization"] = (
        df["credit_utilization_ratio"] >= 0.70
    ).astype(int)

    df["flag_multiple_late_payments"] = (
        df["num_late_payments_last_2yr"] >= 3
    ).astype(int)

    df["flag_excessive_inquiries"] = (
        df["num_hard_inquiries_last_6mo"] >= 4
    ).astype(int)

    df["flag_high_dti"] = (
        df["debt_to_income_ratio"] >= 0.43
    ).astype(int)

    print(
        f"[FeatureEngineer] {len(ENGINEERED_FEATURES)} features added | "
        f"Total columns: {len(df.columns)}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utility: feature summary
# ─────────────────────────────────────────────────────────────────────────────

def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a quick statistical summary of every numeric column.

    Useful for sanity-checking engineered features before model training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame (raw or engineered).

    Returns
    -------
    pd.DataFrame
        Transposed describe table with an extra ``missing_%`` column.
    """
    summary = df.describe().T
    summary["missing_%"] = (df.isnull().sum() / len(df) * 100).round(2)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_generator import generate_credit_dataset, TARGET_COLUMN

    raw_df = generate_credit_dataset(n_samples=500)
    X_raw  = raw_df.drop(columns=[TARGET_COLUMN])
    X_eng  = engineer_features(X_raw)

    print("\nEngineered feature statistics:\n")
    print(feature_summary(X_eng[ENGINEERED_FEATURES]).to_string())
