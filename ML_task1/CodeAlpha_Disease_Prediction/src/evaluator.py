"""
evaluator.py
============
Produces comprehensive evaluation artefacts for one or more fitted models:
  - Classification report (precision, recall, F1, support)
  - ROC-AUC score and ROC curve plot
  - Confusion matrix heatmap
  - Feature importance bar chart (top-20)
  - Side-by-side model comparison table
  - All figures saved to reports/figures/

Design note: Evaluator takes model objects and numpy arrays — it never
touches raw data or file I/O for models. It is safe to call from a
notebook, a training script, or a CI pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)

LABEL_MAP   = {0: "Malignant", 1: "Benign"}
FIGURE_DIR  = Path("reports/figures")
PALETTE     = {"random_forest": "#4F81BD", "xgboost": "#C0504D", "default": "#70AD47"}


class Evaluator:
    """
    Compute and visualise evaluation metrics for binary classification models.

    Parameters
    ----------
    figure_dir : str | Path
        Directory where all plots are saved.
    dpi        : int
        Resolution of saved figures.

    Usage
    -----
    ev = Evaluator()
    ev.evaluate(models, X_test, y_test, feature_names)
    ev.print_comparison()
    """

    def __init__(
        self,
        figure_dir: str = "reports/figures",
        dpi: int = 150,
    ) -> None:
        self.figure_dir = Path(figure_dir)
        self.dpi = dpi
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}

        # Style
        plt.rcParams.update({
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor":   "#FAFAFA",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "font.size": 11,
        })

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        models: Dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "Evaluator":
        """
        Run full evaluation suite for every model in `models`.

        Parameters
        ----------
        models        : dict — {name: fitted_estimator}
        X_test        : np.ndarray — scaled test features
        y_test        : np.ndarray — true binary labels
        feature_names : list[str]  — used for importance plot

        Returns self for chaining.
        """
        logger.info("=" * 60)
        logger.info("EVALUATOR — running evaluation on test set")
        logger.info("=" * 60)

        for name, model in models.items():
            logger.info("Evaluating: %s", name)
            self._evaluate_single(name, model, X_test, y_test, feature_names)

        if len(models) > 1:
            self._plot_roc_comparison(models, X_test, y_test)
            self._plot_pr_comparison(models, X_test, y_test)

        return self

    def print_comparison(self) -> None:
        """Print a formatted comparison table to stdout."""
        if not self.results:
            print("No results yet. Call evaluate() first.")
            return

        rows = []
        for name, metrics in self.results.items():
            rows.append({
                "Model":     name,
                "Accuracy":  f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['macro_precision']:.4f}",
                "Recall":    f"{metrics['macro_recall']:.4f}",
                "F1-Score":  f"{metrics['macro_f1']:.4f}",
                "ROC-AUC":   f"{metrics['roc_auc']:.4f}",
                "Avg Prec":  f"{metrics['avg_precision']:.4f}",
            })

        df = pd.DataFrame(rows).set_index("Model")
        sep = "=" * 75
        print(f"\n{sep}")
        print("MODEL COMPARISON — Test Set Performance")
        print(sep)
        print(df.to_string())
        print(sep)

        best = max(self.results, key=lambda k: self.results[k]["roc_auc"])
        print(f"\nWinner (ROC-AUC): {best} ({self.results[best]['roc_auc']:.4f})\n")

    def get_results_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame of all evaluation metrics."""
        return pd.DataFrame(self.results).T

    # ── Private: single-model evaluation ──────────────────────────────────

    def _evaluate_single(
        self,
        name: str,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]],
    ) -> None:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report  = classification_report(
            y_test, y_pred,
            target_names=["Malignant", "Benign"],
            output_dict=True,
        )
        roc_auc   = roc_auc_score(y_test, y_proba)
        avg_prec  = average_precision_score(y_test, y_proba)
        cm        = confusion_matrix(y_test, y_pred)

        self.results[name] = {
            "accuracy":        report["accuracy"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall":    report["macro avg"]["recall"],
            "macro_f1":        report["macro avg"]["f1-score"],
            "roc_auc":         roc_auc,
            "avg_precision":   avg_prec,
            "report":          report,
            "confusion_matrix": cm,
        }

        # Print full classification report
        print(f"\n{'─'*60}")
        print(f"Classification Report — {name.upper().replace('_',' ')}")
        print('─'*60)
        print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))
        print(f"  ROC-AUC : {roc_auc:.4f}")
        print(f"  Avg Prec: {avg_prec:.4f}")

        # Plots
        self._plot_confusion_matrix(name, cm)
        if feature_names:
            self._plot_feature_importance(name, model, feature_names)

    # ── Private: plot helpers ─────────────────────────────────────────────

    def _plot_confusion_matrix(self, name: str, cm: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Malignant", "Benign"],
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix — {name.replace('_', ' ').title()}", pad=12)
        path = self.figure_dir / f"confusion_matrix_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("Saved: %s", path)

    def _plot_feature_importance(
        self, name: str, model, feature_names: List[str], top_n: int = 20
    ) -> None:
        if not hasattr(model, "feature_importances_"):
            return
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_names  = [feature_names[i] for i in indices]
        top_values = importances[indices]

        color = PALETTE.get(name, PALETTE["default"])
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(range(top_n), top_values[::-1], color=color, alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([n.replace("_", " ") for n in top_names[::-1]], fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_n} Feature Importances — {name.replace('_', ' ').title()}", pad=12)

        # Value labels on bars
        for bar, val in zip(bars, top_values[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

        path = self.figure_dir / f"feature_importance_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("Saved: %s", path)

    def _plot_roc_comparison(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC=0.50)")

        for name, model in models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            color = PALETTE.get(name, PALETTE["default"])
            label = f"{name.replace('_', ' ').title()} (AUC={roc_auc:.4f})"
            ax.plot(fpr, tpr, lw=2, color=color, label=label)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison", pad=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        path = self.figure_dir / "roc_comparison.png"
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("Saved: %s", path)

    def _plot_pr_comparison(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        baseline = np.sum(y_test) / len(y_test)
        ax.axhline(y=baseline, color="k", linestyle="--", lw=1.2,
                   label=f"Baseline (AP={baseline:.2f})")

        for name, model in models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            color = PALETTE.get(name, PALETTE["default"])
            ax.step(recall, precision, lw=2, where="post", color=color,
                    label=f"{name.replace('_', ' ').title()} (AP={ap:.4f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve Comparison", pad=12)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        path = self.figure_dir / "pr_comparison.png"
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("Saved: %s", path)
