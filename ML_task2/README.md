# CodeAlpha_Credit_Scoring_Model

> **CodeAlpha Machine Learning Internship — Task 3**
> Predict an individual's creditworthiness from financial history using a fully modular, lightweight Python ML pipeline.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies: Lightweight](https://img.shields.io/badge/dependencies-lightweight-brightgreen.svg)](#dependencies)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Approach & Methodology](#3-approach--methodology)
4. [Feature Engineering](#4-feature-engineering)
5. [Models](#5-models)
6. [Results](#6-results)
7. [Setup & Installation](#7-setup--installation)
8. [Usage](#8-usage)
9. [Dependencies](#9-dependencies)
10. [Extending the Project](#10-extending-the-project)

---

## 1. Project Overview

Credit scoring is one of the most impactful applications of machine learning in finance. A well-calibrated credit model helps lenders:

- **Reduce default risk** by identifying high-risk applicants early.
- **Expand access to credit** by objectively evaluating borderline applicants.
- **Satisfy regulators** (e.g., CFPB, Basel III) who require explainable, auditable models.

This project builds an end-to-end binary classification pipeline that predicts whether a loan applicant should be **approved** (`1`) or **rejected** (`0`) based on 15 raw financial features, which are expanded to **24 features** through domain-driven feature engineering.

**Key constraints honoured:**
- ✅ Only `pandas`, `numpy`, `scikit-learn`, and `joblib` — total install < 100 MB.
- ✅ No XGBoost, LightGBM, TensorFlow, or PyTorch.
- ✅ Fully reproducible synthetic dataset — runs out of the box with zero external data.

---

## 2. Repository Structure

```
CodeAlpha_Credit_Scoring_Model/
│
├── main.py                     # ← START HERE: end-to-end pipeline orchestrator
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic dataset creation
│   ├── feature_engineer.py     # Domain-driven feature transformations
│   ├── preprocessor.py         # Imputation, scaling, train/val/test split
│   ├── trainer.py              # Logistic Regression & Random Forest training
│   └── evaluator.py            # Metrics, reports, comparison tables
│
├── data/                       # Auto-created — raw CSV written here
│   └── raw_credit_data.csv
│
├── models/                     # Auto-created — serialised models saved here
│   ├── preprocessing_pipeline.joblib
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
│
├── reports/                    # Auto-created — CSV metric reports
│   ├── logistic_regression_metrics.csv
│   ├── logistic_regression_feature_importances.csv
│   ├── random_forest_metrics.csv
│   ├── random_forest_feature_importances.csv
│   └── model_comparison.csv
│
├── requirements.txt
└── README.md
```

---

## 3. Approach & Methodology

```
Raw Data  ──►  Feature Engineering  ──►  Preprocessing  ──►  Training  ──►  Evaluation
   │                   │                       │                  │               │
generate_          engineer_            build_pipeline()    ModelTrainer    evaluate_model()
credit_dataset()   features()           impute + scale      GridSearchCV    compare_models()
```

### 3.1 Data Generation

A synthetic dataset is produced by `src/data_generator.py`. The key design decisions are:

| Decision | Rationale |
|---|---|
| Log-normal income distribution | Matches real-world right-skewed income distributions |
| Logistic approval signal | Ensures features have statistically meaningful relationships with the target |
| ~70 % approval rate | Reflects realistic retail-lending approval rates |
| 2 % random missing values | Forces the pipeline to handle incompleteness robustly |
| Seeded RNG (`seed=42`) | Guarantees identical results across every run |

### 3.2 Data Splitting

Three-way **stratified** split to preserve the class ratio in every partition:

| Split | Size | Purpose |
|---|---|---|
| Training | 68 % | Model fitting and cross-validation |
| Validation | 12 % | Over-fit detection during development |
| Test | 20 % | Final, unbiased performance estimate |

> ⚠ **No leakage**: the preprocessing pipeline is `fit` **only** on training data; validation and test sets are transformed using training-set statistics.

### 3.3 Preprocessing Pipeline

```
Raw features
    │
    ▼
SimpleImputer (strategy='median')   — fills NaN with column median (robust to outliers)
    │
    ▼
StandardScaler                      — zero mean, unit variance
    │
    ▼
Model-ready matrix
```

### 3.4 Hyperparameter Tuning

Both models are tuned with **5-fold stratified `GridSearchCV`**, optimising for **ROC-AUC** — the standard metric for credit scoring because it measures ranking quality independently of the decision threshold.

---

## 4. Feature Engineering

`src/feature_engineer.py` expands the 15 raw columns to **24 features** using domain knowledge:

### Raw Features (15)
| Feature | Description |
|---|---|
| `age` | Applicant age (18–75) |
| `annual_income` | Gross yearly income |
| `monthly_debt_payments` | Total monthly debt obligations |
| `credit_utilization_ratio` | Credit used ÷ total credit limit |
| `num_open_accounts` | Number of active credit accounts |
| `num_late_payments_last_2yr` | Late payments in the past 24 months |
| `num_hard_inquiries_last_6mo` | Hard credit pulls in the past 6 months |
| `oldest_account_age_months` | Age of the oldest credit account |
| `total_credit_limit` | Sum of all credit limits |
| `loan_amount_requested` | Amount the applicant is requesting |
| `employment_length_years` | Years at current employer |
| `has_mortgage` | Binary flag: owns a mortgage (0/1) |
| `has_auto_loan` | Binary flag: has an auto loan (0/1) |
| `debt_to_income_ratio` | Annual debt ÷ annual income |
| `payment_history_score` | Synthetic 0–100 payment quality score |

### Engineered Features (9)
| Feature | Formula / Logic | Financial Interpretation |
|---|---|---|
| `loan_to_income_ratio` | `loan_requested / income` | Stretching-budget indicator |
| `monthly_debt_to_income` | `monthly_debt / (income/12)` | Front-end DTI (lender standard) |
| `credit_limit_to_income` | `credit_limit / income` | Historical credit trust proxy |
| `risk_composite_score` | `0.4·DTI + 0.35·utilisation + 0.25·late_rate` | Weighted multi-factor risk |
| `credit_experience_score` | `(payment_score/100) · log(account_age)` | Quality × length of history |
| `flag_high_utilization` | `utilisation ≥ 70%` | Hard credit-policy threshold |
| `flag_multiple_late_payments` | `late_payments ≥ 3` | Delinquency flag |
| `flag_excessive_inquiries` | `inquiries ≥ 4` | Credit-seeking red flag |
| `flag_high_dti` | `DTI ≥ 43%` | US Qualified Mortgage (QM) threshold |

---

## 5. Models

### Logistic Regression
- **Why:** Industry-standard interpretable baseline. Coefficients map directly to log-odds of approval, satisfying the explainability demands of fair-lending regulations.
- **Regularisation:** L2 (Ridge) via `C` tuning — shrinks coefficients to reduce variance.
- **Class imbalance:** `class_weight='balanced'` automatically up-weights the minority class.

### Random Forest
- **Why:** Captures non-linear interactions and feature interdependencies that a linear model misses. Feature importances provide an additional transparency layer.
- **Ensemble:** Averages 100–200 decision trees trained on bootstrap samples.
- **Class imbalance:** `class_weight='balanced'` applies per-tree.

---

## 6. Results

Results obtained on a 3 000-sample synthetic dataset (600-sample test set):

### Evaluation Metrics — Held-out Test Set

| Model | ROC-AUC | Avg Precision | Precision | Recall | F1-Score | Accuracy | CV AUC | Train Time |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.8704** | **0.9361** | **0.8981** | 0.7818 | 0.8359 | 0.7867 | 0.8955 ± 0.011 | 0.85 s |
| Random Forest | 0.8514 | 0.9239 | 0.8004 | **0.8849** | **0.8405** | 0.7667 | 0.8769 ± 0.014 | 27.9 s |

**🏆 Winner by ROC-AUC: Logistic Regression (0.8704)**

> **Interpretation:** Logistic Regression wins on ROC-AUC and precision (fewer false approvals) while being 33× faster to train. Random Forest wins on recall (catches more creditworthy applicants). The right model depends on the business objective — minimising default losses favours LR; maximising approvals favours RF.

### Top Feature Importances

**Logistic Regression** (coefficient magnitudes):
```
oldest_account_age_months      0.9515  ██████████████████████████████
num_late_payments_last_2yr     0.6702  █████████████████████
risk_composite_score           0.5608  █████████████████
credit_utilization_ratio       0.4745  ██████████████
num_hard_inquiries_last_6mo    0.4086  ████████████
payment_history_score          0.4018  ████████████
```

**Random Forest** (mean impurity decrease):
```
credit_experience_score        0.1202  ██████████████████████████████
risk_composite_score           0.1179  █████████████████████████████
oldest_account_age_months      0.0997  ████████████████████████
payment_history_score          0.0989  ████████████████████████
```

Both models agree that **credit history length**, **payment quality**, and **utilisation** are the most predictive features — consistent with real-world FICO score components.

### Confusion Matrix — Logistic Regression (threshold = 0.50)

```
                   Pred: Rejected   Pred: Approved
True: Rejected           146               37        ← 37 false approvals
True: Approved            91              326         ← 91 missed opportunities
```

---

## 7. Setup & Installation

### Prerequisites

- Python **3.10** or higher
- `pip` (comes with Python)

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/CodeAlpha_Credit_Scoring_Model.git
cd CodeAlpha_Credit_Scoring_Model
```

### Step 2 — (Recommended) Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Total download size is approximately **40–80 MB** depending on your platform — well within GitHub's 100 MB file limit.

---

## 8. Usage

### Run the complete pipeline (recommended)

```bash
python main.py
```

This single command:
1. Generates a synthetic dataset of **5 000** applicants and saves it to `data/`
2. Engineers 9 new domain features (24 total)
3. Splits the data into train / validation / test sets
4. Fits the imputer + scaler pipeline
5. Trains and tunes **Logistic Regression** and **Random Forest** with 5-fold CV
6. Prints detailed evaluation reports to the console
7. Saves CSV metric reports to `reports/`
8. Persists all fitted models to `models/`

### Custom sample count

```bash
python main.py --samples 10000
```

### Skip saving models to disk

```bash
python main.py --no-save
```

### Run individual modules

```bash
# Generate dataset only
python src/data_generator.py

# Feature engineering demo
python src/feature_engineer.py

# Training demo (uses 2 000 samples internally)
python src/trainer.py
```

### Load a saved model for inference

```python
import joblib
import pandas as pd
from src.feature_engineer import engineer_features

# Load artefacts
pipeline = joblib.load("models/preprocessing_pipeline.joblib")
model    = joblib.load("models/logistic_regression.joblib")

# Prepare a new applicant record (raw features only, no target)
applicant = pd.DataFrame([{
    "age": 34,
    "annual_income": 72_000,
    "monthly_debt_payments": 850,
    "credit_utilization_ratio": 0.35,
    "num_open_accounts": 5,
    "num_late_payments_last_2yr": 1,
    "num_hard_inquiries_last_6mo": 2,
    "oldest_account_age_months": 84,
    "total_credit_limit": 28_000,
    "loan_amount_requested": 15_000,
    "employment_length_years": 6,
    "has_mortgage": 1,
    "has_auto_loan": 0,
    "debt_to_income_ratio": 0.142,
    "payment_history_score": 78,
}])

# Engineer features → preprocess → predict
applicant_eng = engineer_features(applicant)
applicant_t   = pipeline.transform(applicant_eng)
approval_prob = model.predict_proba(applicant_t)[0, 1]
decision      = "APPROVED ✅" if approval_prob >= 0.5 else "REJECTED ❌"

print(f"Approval probability: {approval_prob:.1%}")
print(f"Credit decision:      {decision}")
```

---

## 9. Dependencies

| Package | Version | Purpose | Install size |
|---|---|---|---|
| `numpy` | ≥ 1.24 | Numerical arrays & random generation | ~20 MB |
| `pandas` | ≥ 2.0 | DataFrame manipulation | ~30 MB |
| `scikit-learn` | ≥ 1.3 | All ML algorithms, pipelines & metrics | ~30 MB |
| `joblib` | ≥ 1.3 | Model serialisation (bundled with sklearn) | ~1 MB |

**Total: ~80 MB** — well within the 100 MB GitHub file size limit.

No XGBoost, LightGBM, TensorFlow, PyTorch, or any other heavy framework is used or required.

---

## 10. Extending the Project

| Extension | How to Implement |
|---|---|
| **Add a new model** | Subclass or extend `ModelTrainer` in `src/trainer.py`; plug result into `evaluate_model()` |
| **Real dataset** | Replace `generate_credit_dataset()` call in `main.py` with `pd.read_csv("your_data.csv")` |
| **Threshold tuning** | Pass a custom `threshold=` to `evaluate_model()` to optimise Precision/Recall trade-off |
| **SHAP explanations** | Install `shap` (lightweight) and call `shap.TreeExplainer(rf_model)` |
| **Calibration** | Wrap any model in `sklearn.calibration.CalibratedClassifierCV` for better probability estimates |
| **Class imbalance** | Add `SMOTE` from `imbalanced-learn` before the scaling step |
| **More CV folds** | Set `cv_folds=10` when constructing `ModelTrainer()` |

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

Built as part of the **CodeAlpha Machine Learning Internship** programme.
Feature design inspired by FICO® Score components and standard lending industry risk frameworks.
