"""
src/data_generator.py
=====================
Generates a fully synthetic credit-scoring dataset that mirrors realistic
financial behaviour patterns — correlations, skewed distributions, missing
values, and a class-imbalance ratio typical of real lending portfolios.

All randomness is seeded, so the dataset is reproducible across runs.

Author : CodeAlpha Internship Project
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED: int = 42
DEFAULT_N_SAMPLES: int = 5_000
TARGET_COLUMN: str = "credit_approved"

# Raw feature columns produced by the generator (target excluded)
RAW_FEATURE_COLUMNS: list[str] = [
    "age",
    "annual_income",
    "monthly_debt_payments",
    "credit_utilization_ratio",
    "num_open_accounts",
    "num_late_payments_last_2yr",
    "num_hard_inquiries_last_6mo",
    "oldest_account_age_months",
    "total_credit_limit",
    "loan_amount_requested",
    "employment_length_years",
    "has_mortgage",
    "has_auto_loan",
    "debt_to_income_ratio",
    "payment_history_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_credit_dataset(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = RANDOM_SEED,
    output_path: str | None = None,
    missing_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Build and return a synthetic credit-application dataset.

    Each row represents one loan applicant. The binary target
    ``credit_approved`` (1 = approved, 0 = rejected) is derived from a
    calibrated logistic formula so that every feature has a statistically
    meaningful relationship with the outcome.

    Parameters
    ----------
    n_samples : int
        Number of applicant records to create. Default: 5 000.
    random_seed : int
        Seed for NumPy's random generator. Default: 42.
    output_path : str or None
        When supplied, the DataFrame is saved as a CSV at this path before
        being returned.
    missing_rate : float
        Fraction of numeric cells to replace with NaN, simulating real-world
        data incompleteness (default 2 %).

    Returns
    -------
    pd.DataFrame
        Complete dataset including all feature columns and the target column.

    Examples
    --------
    >>> from src.data_generator import generate_credit_dataset
    >>> df = generate_credit_dataset(n_samples=1_000, random_seed=0)
    >>> df.shape
    (1000, 16)
    >>> "credit_approved" in df.columns
    True
    """

    rng = np.random.default_rng(random_seed)

    # ── Demographic & employment ──────────────────────────────────────────────
    age = rng.integers(18, 76, size=n_samples).astype(float)
    employment_length = np.clip(
        rng.exponential(scale=6, size=n_samples), 0, 40
    )

    # ── Income: log-normal → right-skewed, realistic ($15k–$400k) ────────────
    annual_income = np.exp(rng.normal(loc=10.8, scale=0.6, size=n_samples))
    annual_income = np.clip(annual_income, 15_000, 400_000)

    # ── Credit limit positively correlated with income ────────────────────────
    total_credit_limit = np.clip(
        annual_income * rng.uniform(0.5, 3.0, size=n_samples)
        + rng.normal(0, 5_000, size=n_samples),
        1_000,
        500_000,
    )

    # ── Loan amount: lower-income applicants often request proportionally more ─
    loan_amount_requested = np.clip(
        annual_income * rng.uniform(0.05, 0.8, size=n_samples),
        1_000,
        200_000,
    )

    # ── Monthly debt & credit utilisation ────────────────────────────────────
    monthly_debt_payments = np.clip(
        (annual_income / 12) * rng.beta(2, 5, size=n_samples),
        0,
        annual_income / 12,
    )
    credit_utilization_ratio = np.clip(rng.beta(2, 4, size=n_samples), 0, 1)

    # ── Account history ───────────────────────────────────────────────────────
    num_open_accounts       = rng.integers(0, 25, size=n_samples).astype(float)
    oldest_account_age_mths = np.clip(
        rng.exponential(scale=60, size=n_samples) + age * 6, 1, 600
    )

    # ── Negative credit events ────────────────────────────────────────────────
    num_late_payments  = rng.integers(0, 15, size=n_samples).astype(float)
    num_hard_inquiries = rng.integers(0, 10, size=n_samples).astype(float)

    # ── Binary product-ownership flags ────────────────────────────────────────
    has_mortgage  = rng.choice([0, 1], size=n_samples, p=[0.55, 0.45]).astype(float)
    has_auto_loan = rng.choice([0, 1], size=n_samples, p=[0.60, 0.40]).astype(float)

    # ── Derived: debt-to-income ratio ─────────────────────────────────────────
    debt_to_income_ratio = np.clip(
        (monthly_debt_payments * 12) / (annual_income + 1e-9), 0, 2
    )

    # ── Derived: payment history score (0–100) ────────────────────────────────
    #   Penalised by late payments and high utilisation;
    #   boosted by length of credit history.
    payment_history_score = np.clip(
        100
        - (num_late_payments * 7)
        - (credit_utilization_ratio * 30)
        + (oldest_account_age_mths / 60)
        + rng.normal(0, 5, size=n_samples),
        0,
        100,
    )

    # ── Target: credit approval via logistic signal ───────────────────────────
    #   Positive coefficient → raises approval odds
    #   Negative coefficient → lowers approval odds
    log_odds = (
        2.0                                              # baseline intercept
        + 0.030 * (annual_income / 10_000)              # higher income → better
        - 4.000 * debt_to_income_ratio                  # high DTI → worse
        - 3.500 * credit_utilization_ratio              # high utilisation → worse
        + 0.020 * payment_history_score                 # good history → better
        - 0.250 * num_late_payments                     # late payments → worse
        - 0.200 * num_hard_inquiries                    # many inquiries → worse
        + 0.010 * oldest_account_age_mths               # long history → better
        + 0.040 * employment_length                     # stable employment → better
        - 0.010 * (loan_amount_requested / 1_000)       # larger request → riskier
        + rng.normal(0, 0.5, size=n_samples)            # irreducible noise
    )
    approval_prob   = 1.0 / (1.0 + np.exp(-log_odds))
    credit_approved = (rng.uniform(size=n_samples) < approval_prob).astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "age":                         age,
            "annual_income":               annual_income,
            "monthly_debt_payments":       monthly_debt_payments,
            "credit_utilization_ratio":    credit_utilization_ratio,
            "num_open_accounts":           num_open_accounts,
            "num_late_payments_last_2yr":  num_late_payments,
            "num_hard_inquiries_last_6mo": num_hard_inquiries,
            "oldest_account_age_months":   oldest_account_age_mths,
            "total_credit_limit":          total_credit_limit,
            "loan_amount_requested":       loan_amount_requested,
            "employment_length_years":     employment_length,
            "has_mortgage":                has_mortgage,
            "has_auto_loan":               has_auto_loan,
            "debt_to_income_ratio":        debt_to_income_ratio,
            "payment_history_score":       payment_history_score,
            TARGET_COLUMN:                 credit_approved,
        }
    )

    # ── Inject realistic missing values (target is never corrupted) ───────────
    _introduce_missing(df, rng, missing_rate)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[DataGenerator] Saved → {output_path}")

    approval_rate = df[TARGET_COLUMN].mean()
    print(
        f"[DataGenerator] {len(df):,} samples generated | "
        f"Approval rate: {approval_rate:.1%} | "
        f"Features: {len(df.columns) - 1}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _introduce_missing(
    df: pd.DataFrame,
    rng: np.random.Generator,
    missing_rate: float,
) -> None:
    """
    Randomly replace *missing_rate* fraction of numeric cells with NaN.
    Modifies *df* **in-place**. The target column is never corrupted.
    """
    cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != TARGET_COLUMN
    ]
    for col in cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    data = generate_credit_dataset(output_path="data/raw_credit_data.csv")
    print("\nSample statistics:\n")
    print(data.describe().T.to_string())
