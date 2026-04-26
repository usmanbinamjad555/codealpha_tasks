"""
data_generator.py
=================
Generates a synthetic credit scoring dataset that mimics real-world
financial behaviour patterns. All distributions and correlations are
designed to reflect plausible lending scenarios.

Author : CodeAlpha Internship Project
License: MIT
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
DEFAULT_N_SAMPLES = 5_000

FEATURE_COLUMNS = [
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
    "debt_to_income_ratio",      # engineered inside generator for realism
    "payment_history_score",     # synthetic 0-100 score
]

TARGET_COLUMN = "credit_approved"


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_credit_dataset(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = RANDOM_SEED,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic credit-application dataset.

    Each row represents one loan applicant.  The binary target
    ``credit_approved`` (1 = approved, 0 = rejected) is derived from a
    weighted logistic formula so that high-risk features meaningfully
    reduce approval probability.

    Parameters
    ----------
    n_samples : int
        Number of applicant records to create. Default: 5 000.
    random_seed : int
        NumPy random seed for reproducibility. Default: 42.
    output_path : str or None
        If provided, the DataFrame is saved as a CSV at this path.

    Returns
    -------
    pd.DataFrame
        Full dataset including features and the target column.

    Example
    -------
    >>> from data_generator import generate_credit_dataset
    >>> df = generate_credit_dataset(n_samples=1_000)
    >>> df.shape
    (1000, 16)
    """

    rng = np.random.default_rng(random_seed)

    # ── Demographic & employment ──────────────────────────────────────────────
    age = rng.integers(18, 76, size=n_samples).astype(float)
    employment_length = np.clip(rng.exponential(scale=6, size=n_samples), 0, 40)

    # ── Income (log-normal → right-skewed, realistic) ─────────────────────────
    annual_income = np.exp(rng.normal(loc=10.8, scale=0.6, size=n_samples))
    annual_income = np.clip(annual_income, 15_000, 400_000)

    # ── Credit limit positively correlated with income ─────────────────────────
    total_credit_limit = (
        annual_income * rng.uniform(0.5, 3.0, size=n_samples)
        + rng.normal(0, 5_000, size=n_samples)
    )
    total_credit_limit = np.clip(total_credit_limit, 1_000, 500_000)

    # ── Loan request (people with lower income request higher relative amounts) ──
    loan_amount_requested = np.clip(
        annual_income * rng.uniform(0.05, 0.8, size=n_samples),
        1_000,
        200_000,
    )

    # ── Monthly debt & utilisation ─────────────────────────────────────────────
    monthly_debt_payments = np.clip(
        (annual_income / 12) * rng.beta(2, 5, size=n_samples),
        0,
        annual_income / 12,
    )
    credit_utilization_ratio = np.clip(rng.beta(2, 4, size=n_samples), 0, 1)

    # ── Account history ────────────────────────────────────────────────────────
    num_open_accounts = rng.integers(0, 25, size=n_samples).astype(float)
    oldest_account_age_months = np.clip(
        rng.exponential(scale=60, size=n_samples) + age * 6,
        1,
        600,
    )

    # ── Negative credit events ─────────────────────────────────────────────────
    num_late_payments = rng.integers(0, 15, size=n_samples).astype(float)
    num_hard_inquiries = rng.integers(0, 10, size=n_samples).astype(float)

    # ── Binary flags ───────────────────────────────────────────────────────────
    has_mortgage = rng.choice([0, 1], size=n_samples, p=[0.55, 0.45]).astype(float)
    has_auto_loan = rng.choice([0, 1], size=n_samples, p=[0.60, 0.40]).astype(float)

    # ── Derived features ───────────────────────────────────────────────────────
    debt_to_income_ratio = np.clip(
        (monthly_debt_payments * 12) / (annual_income + 1e-6), 0, 2
    )

    # Payment history score: penalised by late payments and high utilisation
    payment_history_score = np.clip(
        100
        - (num_late_payments * 7)
        - (credit_utilization_ratio * 30)
        + (oldest_account_age_months / 60)
        + rng.normal(0, 5, size=n_samples),
        0,
        100,
    )

    # ── Approval signal (logistic) ────────────────────────────────────────────
    # Positive weights → increases approval odds
    # Negative weights → decreases approval odds
    log_odds = (
        2.0                                                    # baseline intercept
        + 0.03  * (annual_income / 10_000)                    # higher income = better
        - 4.0   * debt_to_income_ratio                        # high DTI = worse
        - 3.5   * credit_utilization_ratio                    # high util = worse
        + 0.02  * payment_history_score                       # good history = better
        - 0.25  * num_late_payments                           # late pays = worse
        - 0.20  * num_hard_inquiries                          # many inquiries = worse
        + 0.01  * oldest_account_age_months                   # long history = better
        + 0.04  * employment_length                           # stable job = better
        - 0.01  * (loan_amount_requested / 1_000)             # larger ask = riskier
        + rng.normal(0, 0.5, size=n_samples)                  # noise
    )
    approval_prob = 1 / (1 + np.exp(-log_odds))
    credit_approved = (rng.uniform(size=n_samples) < approval_prob).astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "age":                        age,
            "annual_income":              annual_income,
            "monthly_debt_payments":      monthly_debt_payments,
            "credit_utilization_ratio":   credit_utilization_ratio,
            "num_open_accounts":          num_open_accounts,
            "num_late_payments_last_2yr": num_late_payments,
            "num_hard_inquiries_last_6mo": num_hard_inquiries,
            "oldest_account_age_months":  oldest_account_age_months,
            "total_credit_limit":         total_credit_limit,
            "loan_amount_requested":      loan_amount_requested,
            "employment_length_years":    employment_length,
            "has_mortgage":               has_mortgage,
            "has_auto_loan":              has_auto_loan,
            "debt_to_income_ratio":       debt_to_income_ratio,
            "payment_history_score":      payment_history_score,
            TARGET_COLUMN:                credit_approved,
        }
    )

    # ── Introduce a small fraction of missing values (realistic) ─────────────
    _introduce_missing(df, rng, missing_rate=0.02)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[DataGenerator] Dataset saved → {output_path}  ({len(df):,} rows)")

    print(
        f"[DataGenerator] Generated {len(df):,} samples | "
        f"Approval rate: {df[TARGET_COLUMN].mean():.1%}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _introduce_missing(
    df: pd.DataFrame,
    rng: np.random.Generator,
    missing_rate: float = 0.02,
) -> None:
    """
    Randomly replace a small fraction of numeric values with NaN to simulate
    real-world data incompleteness.  Operates **in-place**.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to corrupt.
    rng : np.random.Generator
        Seeded random generator.
    missing_rate : float
        Fraction of cells to set to NaN (default 2 %).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Never corrupt the target column
    numeric_cols = [c for c in numeric_cols if c != TARGET_COLUMN]

    for col in numeric_cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_credit_dataset(output_path="data/raw_credit_data.csv")
    print(df.describe().T.to_string())
