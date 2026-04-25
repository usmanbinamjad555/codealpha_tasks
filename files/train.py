"""
train.py — Master Training Script
===================================
Orchestrates the full ML pipeline end-to-end:

    DataPipeline  →  ModelTrainer  →  Evaluator  →  Artefact saving

Run from the project root:
    python train.py               # with hyperparameter tuning  (default)
    python train.py --no-tune     # skip tuning, use preset params (faster)

Saved artefacts
---------------
    models/best_model.joblib      — best-performing model
    models/random_forest.joblib   — Random Forest model
    models/xgboost.joblib         — XGBoost model
    models/scaler.joblib          — fitted scaler + feature names bundle
    reports/figures/              — all evaluation plots
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import DataPipeline
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disease Prediction — Training Script")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter search and use default model parameters.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of RandomizedSearchCV iterations per model (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   CodeAlpha Disease Prediction — Training    ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Hyperparameter tuning: %s", not args.no_tune)

    # ── 1. Data Pipeline ──────────────────────────────────────────────────
    pipeline = DataPipeline().run()

    logger.info("\nClass distribution:")
    print(pipeline.get_class_distribution().to_string())

    # ── 2. Model Training ─────────────────────────────────────────────────
    trainer = ModelTrainer(
        tune_hyperparams=not args.no_tune,
        n_iter_search=args.n_iter,
    ).train(
        pipeline.X_train, pipeline.y_train,
        pipeline.X_val,   pipeline.y_val,
    )

    # ── 3. Evaluation on held-out test set ────────────────────────────────
    evaluator = Evaluator(figure_dir="reports/figures")
    evaluator.evaluate(
        models        = trainer.models,
        X_test        = pipeline.X_test,
        y_test        = pipeline.y_test,
        feature_names = pipeline.feature_names,
    )
    evaluator.print_comparison()

    # ── 4. Save artefacts ─────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)

    pipeline.save_scaler("models/scaler.joblib")

    for name in trainer.models:
        trainer.save_model(name, f"models/{name}.joblib")

    trainer.save_best("models/best_model.joblib")

    elapsed = time.time() - t_start
    logger.info("\n✓ Training complete in %.1fs", elapsed)
    logger.info("  Best model : %s", trainer.best_model_name)
    logger.info("  Artefacts  : models/  reports/figures/")
    logger.info("\nTo run inference:")
    logger.info("  python predict.py --help")


if __name__ == "__main__":
    main()
