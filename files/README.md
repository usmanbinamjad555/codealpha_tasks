# CodeAlpha_Disease_Prediction

> **CodeAlpha Machine Learning Internship — Task 3**
> Predict the possibility of breast cancer (Benign / Malignant) from patient diagnostic measurements using an end-to-end modular ML pipeline.

---

## Results (Test Set — 114 samples)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 95.61 % | 0.9510 | 0.9554 | 0.9531 | **0.9950** |
| **XGBoost** ✓ | **97.37 %** | **0.9697** | **0.9742** | **0.9719** | **0.9954** |

XGBoost is automatically selected as the deployment model.

---

## Project Structure

```
CodeAlpha_Disease_Prediction/
│
├── src/                         # Core library — import these in your API
│   ├── __init__.py              # Package exports
│   ├── data_pipeline.py         # Load → clean → engineer → split → scale
│   ├── model_trainer.py         # Train & tune Random Forest + XGBoost
│   ├── evaluator.py             # Metrics, plots, comparison reports
│   └── predictor.py             # Decoupled inference engine (API-ready)
│
├── models/                      # Saved artefacts (created by train.py)
│   ├── best_model.joblib        # Winning model
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── scaler.joblib            # Scaler + feature name bundle
│
├── reports/
│   └── figures/                 # Auto-generated evaluation plots
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_xgboost.png
│       ├── feature_importance_random_forest.png
│       ├── feature_importance_xgboost.png
│       ├── roc_comparison.png
│       └── pr_comparison.png
│
├── train.py                     # Master training script
├── predict.py                   # CLI inference demo
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CodeAlpha_Disease_Prediction.git
cd CodeAlpha_Disease_Prediction
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train both models

```bash
# With hyperparameter tuning (~2-3 min, best accuracy)
python train.py

# Without tuning (fast, ~5 seconds, still >95% accuracy)
python train.py --no-tune
```

**Training output includes:**
- Class distribution table for all splits
- Validation ROC-AUC during training
- Full classification report per model on the test set
- Side-by-side model comparison table
- All figures saved to `reports/figures/`

### Run inference

```bash
# Demo: predict on two hard-coded sample patients
python predict.py

# Use a specific model
python predict.py --model models/random_forest.joblib

# Batch prediction from a JSON file
python predict.py --batch my_patients.json
```

**Sample batch JSON format:**

```json
[
  {
    "mean radius": 12.46, "mean texture": 24.04, "mean perimeter": 83.97,
    "mean area": 475.9, "mean smoothness": 0.1186, "mean compactness": 0.2396,
    "mean concavity": 0.2273, "mean concave points": 0.08543,
    "mean symmetry": 0.203, "mean fractal dimension": 0.08243,
    "radius error": 0.2976, "texture error": 1.599, "perimeter error": 2.039,
    "area error": 23.94, "smoothness error": 0.007149, "compactness error": 0.07217,
    "concavity error": 0.07743, "concave points error": 0.01432,
    "symmetry error": 0.01789, "fractal dimension error": 0.01263,
    "worst radius": 13.36, "worst texture": 29.25, "worst perimeter": 93.11,
    "worst area": 554.9, "worst smoothness": 0.1675, "worst compactness": 0.6153,
    "worst concavity": 0.6189, "worst concave points": 0.1848,
    "worst symmetry": 0.3748, "worst fractal dimension": 0.1547
  }
]
```

---

## Approach

### Dataset
UCI Breast Cancer Wisconsin (Diagnostic) dataset loaded directly from `sklearn.datasets`.
- **569 samples** — 357 Benign, 212 Malignant
- **30 numeric features** computed from digitised FNA (fine needle aspirate) images
- Splits: 68% train / 12% validation / 20% test (all stratified)

### Data Pipeline (`src/data_pipeline.py`)

| Stage | Details |
|---|---|
| Validation | Checks for missing values, negatives, duplicates |
| Feature engineering | 4 domain-ratio features + 6 log1p transforms = 40 total features |
| Scaling | `StandardScaler` fit **only on training data** — no leakage |

**Engineered features:**
- `compactness_ratio` — mean compactness / mean area (density proxy)
- `perimeter_radius_ratio` — mean perimeter / mean radius (shape regularity)
- `concavity_symmetry_ratio` — mean concavity / mean symmetry (asymmetry severity)
- `worst_mean_area_ratio` — worst area / mean area (tumour growth spread)
- `log_*` — log1p of right-skewed area and perimeter columns

### Models (`src/model_trainer.py`)

Both models use `RandomizedSearchCV` with 5-fold stratified cross-validation on the training set. `scale_pos_weight` is computed automatically to handle class imbalance in XGBoost.

| Model | Key Hyperparameters Tuned |
|---|---|
| Random Forest | n_estimators, max_depth, min_samples_split, max_features, class_weight |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha/lambda |

### Evaluation (`src/evaluator.py`)

- Classification report (precision, recall, F1, support per class)
- ROC-AUC and Average Precision scores
- Confusion matrix heatmap
- Top-20 feature importance bar chart
- ROC curve overlay comparison
- Precision-Recall curve overlay comparison

---

## Backend API Integration

The `Predictor` class in `src/predictor.py` is fully decoupled from all training code. A FastAPI integration requires only:

```python
# pip install fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import Predictor

app = FastAPI(title="Disease Prediction API")

predictor = Predictor.from_files(
    model_path  = "models/best_model.joblib",
    scaler_path = "models/scaler.joblib",
)

class PatientFeatures(BaseModel):
    mean_radius: float
    mean_texture: float
    # ... all 30 UCI features

@app.post("/predict")
def predict(patient: PatientFeatures):
    # Map snake_case back to UCI space-separated names
    data = {k.replace("_", " "): v for k, v in patient.model_dump().items()}
    result = predictor.predict(data)
    return result.to_dict()
```

Run with: `uvicorn api:app --reload`

---

## Disclaimer

This project is for **educational purposes only**. It must not be used to inform real clinical decisions. Always consult a qualified medical professional for diagnosis.

---

*CodeAlpha Machine Learning Internship*
