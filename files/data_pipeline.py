"""
data_pipeline.py
================
Handles every data-facing concern:
  - Loading the UCI Breast Cancer dataset
  - Cleaning and validation
  - Feature engineering (domain-inspired ratios + log transforms)
  - Train / validation / test splitting with stratification
  - Fitting and persisting a StandardScaler

The fitted scaler is stored as an attribute so that the Predictor can
reload it independently of any training code, keeping inference fully
decoupled from the training pipeline.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15
LABEL_MAP    = {0: "Malignant", 1: "Benign"}

LOG_TRANSFORM_FEATURES = [
    "mean area", "mean perimeter",
    "worst area", "worst perimeter",
    "area error", "perimeter error",
]


class DataPipeline:
    """
    End-to-end data preparation for breast cancer classification.

    Parameters
    ----------
    test_size     : float — fraction of full dataset for hold-out test set
    val_size      : float — fraction of training data for validation set
    random_state  : int   — global reproducibility seed

    Attributes (populated after .run())
    ------------------------------------
    raw_df, processed_df : pd.DataFrame
    feature_names        : list[str]
    scaler               : StandardScaler (fit on training data only)
    X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
    """

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        val_size: float = VAL_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.feature_names: list = []
        self.scaler: Optional[StandardScaler] = None

        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> "DataPipeline":
        """Execute: load → validate → engineer → split → scale. Returns self."""
        logger.info("=" * 60)
        logger.info("DATA PIPELINE — starting")
        logger.info("=" * 60)
        self._load()
        self._validate()
        self._engineer_features()
        self._split()
        self._scale()
        logger.info(
            "PIPELINE COMPLETE | train=%d  val=%d  test=%d",
            len(self.y_train), len(self.y_val), len(self.y_test),
        )
        return self

    def save_scaler(self, path: str) -> None:
        """Persist fitted scaler + feature names to disk for the Predictor."""
        if self.scaler is None:
            raise RuntimeError("Pipeline not run. Call .run() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, path)
        logger.info("Scaler saved → %s", path)

    def get_class_distribution(self) -> pd.DataFrame:
        """Return class balance summary across all three splits."""
        rows = []
        for name, y in [("train", self.y_train), ("val", self.y_val), ("test", self.y_test)]:
            counts = pd.Series(y).value_counts().sort_index()
            rows.append({
                "split": name,
                "Malignant (0)": counts.get(0, 0),
                "Benign (1)":    counts.get(1, 0),
                "total": len(y),
            })
        return pd.DataFrame(rows).set_index("split")

    # ── Private helpers ────────────────────────────────────────────────────

    def _load(self) -> None:
        logger.info("Loading UCI Breast Cancer dataset ...")
        data = load_breast_cancer(as_frame=True)
        self.raw_df = data.frame.copy()
        self.raw_df.rename(columns={"target": "diagnosis"}, inplace=True)
        logger.info("Loaded %d samples x %d features", len(self.raw_df), self.raw_df.shape[1] - 1)
        logger.info("Class balance -> %s", dict(self.raw_df["diagnosis"].value_counts()))

    def _validate(self) -> None:
        logger.info("Validating data ...")
        missing = self.raw_df.isnull().sum().sum()
        if missing > 0:
            logger.warning("Found %d missing values — forward-filling.", missing)
            self.raw_df.ffill(inplace=True)
        else:
            logger.info("No missing values found")

        feature_cols = [c for c in self.raw_df.columns if c != "diagnosis"]
        neg = (self.raw_df[feature_cols] < 0).sum().sum()
        if neg > 0:
            logger.warning("%d negative values in positive-only features. Clipping to 0.", neg)
            self.raw_df[feature_cols] = self.raw_df[feature_cols].clip(lower=0)
        else:
            logger.info("No invalid negative measurements")

        dupes = self.raw_df.duplicated().sum()
        if dupes > 0:
            logger.warning("Dropping %d duplicate rows.", dupes)
            self.raw_df.drop_duplicates(inplace=True)

    def _engineer_features(self) -> None:
        """
        Create domain-informed features from the 30 raw measurements.

        Engineered features
        --------------------
        compactness_ratio        : mean compactness / mean area   (density proxy)
        perimeter_radius_ratio   : mean perimeter / mean radius   (shape regularity)
        concavity_symmetry_ratio : mean concavity / mean symmetry (asymmetry severity)
        worst_mean_area_ratio    : worst area / mean area         (growth spread)
        log_<feature>            : log1p of right-skewed area/perimeter cols
        """
        logger.info("Engineering features ...")
        df = self.raw_df.copy()
        eps = 1e-8

        df["compactness_ratio"]        = df["mean compactness"] / (df["mean area"] + eps)
        df["perimeter_radius_ratio"]   = df["mean perimeter"]   / (df["mean radius"] + eps)
        df["concavity_symmetry_ratio"] = df["mean concavity"]   / (df["mean symmetry"] + eps)
        df["worst_mean_area_ratio"]    = df["worst area"]       / (df["mean area"] + eps)

        for col in LOG_TRANSFORM_FEATURES:
            if col in df.columns:
                df[f"log_{col.replace(' ', '_')}"] = np.log1p(df[col])

        self.processed_df = df
        self.feature_names = [c for c in df.columns if c != "diagnosis"]
        logger.info(
            "Feature matrix: %d -> %d columns after engineering",
            len(self.raw_df.columns) - 1,
            len(self.feature_names),
        )

    def _split(self) -> None:
        logger.info("Splitting data (stratified) ...")
        X = self.processed_df[self.feature_names].values
        y = self.processed_df["diagnosis"].values

        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=self.val_size, stratify=y_tv, random_state=self.random_state
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test

    def _scale(self) -> None:
        logger.info("Scaling features (StandardScaler fit on train only) ...")
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val   = self.scaler.transform(self.X_val)
        self.X_test  = self.scaler.transform(self.X_test)
