"""
predict.py — CLI Inference Script
====================================
Runs the decoupled Predictor on sample patient data.

Usage
-----
    # Predict using the best saved model
    python predict.py

    # Predict using a specific model
    python predict.py --model models/random_forest.joblib

    # Run batch prediction from a JSON file
    python predict.py --batch sample_patients.json

This script is intentionally thin — all logic lives in src/predictor.py,
which is what a FastAPI/Flask app would import directly.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Sample patient data (real values from the UCI dataset) ─────────────────────
SAMPLE_BENIGN = {
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
    "worst symmetry": 0.3748, "worst fractal dimension": 0.1547,
}

SAMPLE_MALIGNANT = {
    "mean radius": 20.57, "mean texture": 17.77, "mean perimeter": 132.9,
    "mean area": 1326.0, "mean smoothness": 0.08474, "mean compactness": 0.07864,
    "mean concavity": 0.0869, "mean concave points": 0.07017,
    "mean symmetry": 0.1812, "mean fractal dimension": 0.05667,
    "radius error": 0.5435, "texture error": 0.7339, "perimeter error": 3.398,
    "area error": 74.08, "smoothness error": 0.005225, "compactness error": 0.01308,
    "concavity error": 0.0186, "concave points error": 0.0134,
    "symmetry error": 0.01389, "fractal dimension error": 0.003532,
    "worst radius": 24.99, "worst texture": 23.41, "worst perimeter": 158.8,
    "worst area": 1956.0, "worst smoothness": 0.1238, "worst compactness": 0.1866,
    "worst concavity": 0.2416, "worst concave points": 0.186,
    "worst symmetry": 0.275, "worst fractal dimension": 0.08902,
}


def print_result(result, label: str) -> None:
    """Pretty-print a PredictionResult."""
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"Patient: {label}")
    print(sep)
    d = result.to_dict()
    emoji = "✅" if d["label"] == "Benign" else "⚠️ "
    print(f"  Prediction  : {emoji}  {d['label']}")
    print(f"  Confidence  : {d['confidence'].upper()}")
    print(f"  P(Benign)   : {d['probability_benign']:.4f}  ({d['probability_benign']*100:.1f}%)")
    print(f"  P(Malignant): {d['probability_malignant']:.4f}  ({d['probability_malignant']*100:.1f}%)")
    print(f"  Model       : {d['model_name']}")
    if d["warnings"]:
        print(f"  Warnings    : {'; '.join(d['warnings'])}")
    print(sep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Disease Prediction — Inference CLI")
    p.add_argument("--model",  default="models/best_model.joblib", help="Path to model .joblib")
    p.add_argument("--scaler", default="models/scaler.joblib",     help="Path to scaler .joblib")
    p.add_argument("--batch",  default=None, help="Path to JSON file with list of patient dicts")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Check artefacts exist
    for path_str in [args.model, args.scaler]:
        if not Path(path_str).exists():
            logger.error("File not found: %s  — Run 'python train.py' first.", path_str)
            sys.exit(1)

    predictor = Predictor.from_files(model_path=args.model, scaler_path=args.scaler)

    if args.batch:
        # Batch inference from JSON file
        with open(args.batch) as f:
            patients = json.load(f)
        results = predictor.predict_batch(patients)
        for i, result in enumerate(results):
            print_result(result, f"Patient #{i+1}")
    else:
        # Demo: run both sample patients
        print("\n" + "═" * 50)
        print("DEMO INFERENCE — two sample patients")
        print("═" * 50)
        result_b = predictor.predict(SAMPLE_BENIGN)
        result_m = predictor.predict(SAMPLE_MALIGNANT)
        print_result(result_b, "Sample Benign Patient")
        print_result(result_m, "Sample Malignant Patient")


if __name__ == "__main__":
    main()
