"""
predictor.py
============
Decoupled inference engine — the ONLY module that a backend API needs
to import. It has zero dependency on DataPipeline or ModelTrainer.

Responsibilities
----------------
  1. Load a serialised model (joblib) and scaler artefacts from disk.
  2. Accept a dict (single patient) or list[dict] (batch) of raw feature
     values as input.
  3. Apply the saved feature-engineering transformations and scaler.
  4. Return a structured PredictionResult dataclass.

FastAPI integration example (see README for full snippet)
----------------------------------------------------------
    from src.predictor import Predictor

    predictor = Predictor.from_files(
        model_path  = "models/best_model.joblib",
        scaler_path = "models/scaler.joblib",
    )

    @app.post("/predict")
    def predict(data: PatientFeatures):
        return predictor.predict(data.model_dump())
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "Malignant", 1: "Benign"}

# Right-skewed features that were log1p-transformed during training
LOG_FEATURES = [
    "mean area", "mean perimeter",
    "worst area", "worst perimeter",
    "area error", "perimeter error",
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Structured output from a single inference call.

    Fields
    ------
    label          : "Benign" or "Malignant"
    label_code     : 0 (Malignant) or 1 (Benign)
    probability_benign    : P(Benign)  — float in [0, 1]
    probability_malignant : P(Malignant) — float in [0, 1]
    confidence     : "high" | "medium" | "low" based on max probability
    model_name     : name of the loaded model
    warnings       : list of any data quality warnings detected at inference
    """
    label:                 str
    label_code:            int
    probability_benign:    float
    probability_malignant: float
    confidence:            str
    model_name:            str
    warnings:              List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":                 self.label,
            "label_code":            self.label_code,
            "probability_benign":    round(self.probability_benign, 4),
            "probability_malignant": round(self.probability_malignant, 4),
            "confidence":            self.confidence,
            "model_name":            self.model_name,
            "warnings":              self.warnings,
        }


# ── Predictor ─────────────────────────────────────────────────────────────────

class Predictor:
    """
    Load a trained model + scaler and serve real-time predictions.

    This class intentionally knows NOTHING about sklearn datasets, training
    loops, or evaluation. It is the only artefact your API server imports.

    Parameters
    ----------
    model        : fitted sklearn/XGBoost estimator with predict_proba
    scaler       : fitted StandardScaler
    feature_names: ordered list of feature names expected by the model
    model_name   : human-readable name for logging / response metadata
    """

    def __init__(
        self,
        model,
        scaler,
        feature_names: List[str],
        model_name: str = "unknown",
    ) -> None:
        self.model         = model
        self.scaler        = scaler
        self.feature_names = feature_names
        self.model_name    = model_name

    # ── Factory constructors ───────────────────────────────────────────────

    @classmethod
    def from_files(cls, model_path: str, scaler_path: str) -> "Predictor":
        """
        Load model and scaler from disk.

        Parameters
        ----------
        model_path  : path to joblib model file (e.g. "models/best_model.joblib")
        scaler_path : path to joblib scaler bundle (e.g. "models/scaler.joblib")
        """
        model_path  = Path(model_path)
        scaler_path = Path(scaler_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        model       = joblib.load(model_path)
        bundle      = joblib.load(scaler_path)
        scaler      = bundle["scaler"]
        feat_names  = bundle["feature_names"]

        logger.info("Predictor loaded | model=%s | features=%d", model_path.stem, len(feat_names))
        return cls(model=model, scaler=scaler, feature_names=feat_names, model_name=model_path.stem)

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, patient_data: Dict[str, float]) -> PredictionResult:
        """
        Predict diagnosis for a single patient.

        Parameters
        ----------
        patient_data : dict mapping feature_name -> raw (unscaled) value.
                       Must contain all 30 base UCI features.
                       Engineered features are computed internally.

        Returns
        -------
        PredictionResult dataclass (call .to_dict() for JSON serialisation)
        """
        warnings_list: List[str] = []

        # ── 1. Validate & collect raw features ──────────────────────────
        raw = self._validate_input(patient_data, warnings_list)

        # ── 2. Apply identical feature engineering from training ─────────
        engineered = self._engineer(raw, warnings_list)

        # ── 3. Build feature vector in the exact column order the model
        #       was trained on ─────────────────────────────────────────
        vector = self._build_vector(engineered, warnings_list)

        # ── 4. Scale using the saved scaler ──────────────────────────────
        vector_scaled = self.scaler.transform(vector.reshape(1, -1))

        # ── 5. Predict ────────────────────────────────────────────────────
        label_code = int(self.model.predict(vector_scaled)[0])
        proba      = self.model.predict_proba(vector_scaled)[0]
        p_malignant, p_benign = float(proba[0]), float(proba[1])
        max_conf   = max(p_malignant, p_benign)

        return PredictionResult(
            label                 = LABEL_MAP[label_code],
            label_code            = label_code,
            probability_benign    = p_benign,
            probability_malignant = p_malignant,
            confidence            = self._confidence_level(max_conf),
            model_name            = self.model_name,
            warnings              = warnings_list,
        )

    def predict_batch(
        self, patient_list: List[Dict[str, float]]
    ) -> List[PredictionResult]:
        """
        Predict diagnoses for a list of patients.
        Each dict follows the same schema as predict().
        """
        return [self.predict(p) for p in patient_list]

    # ── Private helpers ────────────────────────────────────────────────────

    def _validate_input(
        self, data: Dict[str, float], warnings_list: List[str]
    ) -> Dict[str, float]:
        """Check for missing / negative values. Returns cleaned dict."""
        # Base UCI features (no engineered ones)
        base_features = [
            f for f in self.feature_names
            if not f.startswith("log_")
            and f not in (
                "compactness_ratio", "perimeter_radius_ratio",
                "concavity_symmetry_ratio", "worst_mean_area_ratio"
            )
        ]
        raw = {}
        for feat in base_features:
            val = data.get(feat)
            if val is None:
                warnings_list.append(f"Missing feature '{feat}' — using 0.0")
                val = 0.0
            if val < 0:
                warnings_list.append(f"Negative value for '{feat}' ({val}) — clipping to 0.0")
                val = 0.0
            raw[feat] = float(val)
        return raw

    def _engineer(
        self, raw: Dict[str, float], warnings_list: List[str]
    ) -> Dict[str, float]:
        """Mirror the feature engineering applied during training."""
        eps = 1e-8
        d = dict(raw)

        # Ratio features
        d["compactness_ratio"]        = d.get("mean compactness", 0) / (d.get("mean area", eps) + eps)
        d["perimeter_radius_ratio"]   = d.get("mean perimeter", 0)   / (d.get("mean radius", eps) + eps)
        d["concavity_symmetry_ratio"] = d.get("mean concavity", 0)   / (d.get("mean symmetry", eps) + eps)
        d["worst_mean_area_ratio"]    = d.get("worst area", 0)       / (d.get("mean area", eps) + eps)

        # Log transforms
        for col in LOG_FEATURES:
            key = f"log_{col.replace(' ', '_')}"
            d[key] = float(np.log1p(d.get(col, 0)))

        return d

    def _build_vector(
        self, engineered: Dict[str, float], warnings_list: List[str]
    ) -> np.ndarray:
        """Assemble a float array in the exact column order used during training."""
        vector = []
        for feat in self.feature_names:
            val = engineered.get(feat, 0.0)
            if feat not in engineered:
                warnings_list.append(f"Engineered feature '{feat}' not found — using 0.0")
            vector.append(val)
        return np.array(vector, dtype=np.float64)

    @staticmethod
    def _confidence_level(max_prob: float) -> str:
        if max_prob >= 0.85:
            return "high"
        if max_prob >= 0.65:
            return "medium"
        return "low"
