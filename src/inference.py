from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from xgboost import XGBClassifier

from src.config import ROOT, load_params

logger = logging.getLogger(__name__)

# Mapping interne → API (2,4,6) → (0,1,2)
INTERNAL_TO_API_LABELS = {2: 0, 4: 1, 6: 2}
API_TO_INTERNAL_LABELS = {0: 2, 1: 4, 2: 6}


def _artifacts_dir() -> Path:
    return ROOT / "models" / "artifacts"


def _processed_dir() -> Path:
    params = load_params()
    return ROOT / params["data"]["processed_dir"]


def load_label_encoder():
    return joblib.load(_processed_dir() / "label_encoder.joblib")


def load_best_model() -> tuple[Any, Any]:
    """Charge le meilleur modèle (nom issu de best_model.json) et le LabelEncoder."""
    art = _artifacts_dir()
    with open(art / "best_model.json", encoding="utf-8") as f:
        best = json.load(f)
    name = str(best["model_name"])
    le = load_label_encoder()

    if name == "logistic_regression":
        return joblib.load(art / "logreg.joblib"), le
    if name == "random_forest":
        return joblib.load(art / "rf.joblib"), le
    if name == "xgboost":
        path = art / "xgb.joblib"
        if not path.is_file():
            path = art / "xgb.json"  # rétrocompatibilité anciens artefacts
            clf = XGBClassifier()
            clf.load_model(path)
            return clf, le
        return joblib.load(path), le
    raise ValueError(f"Modèle inconnu: {name}")


def predict_batch(model, X_flat: np.ndarray, le) -> tuple[np.ndarray, np.ndarray]:
    """X_flat: (n, n_pixels) normalisé comme à l'entraînement."""
    proba = model.predict_proba(X_flat)
    pred_idx = np.argmax(proba, axis=1)
    return pred_idx, proba


def indices_to_labels(indices: np.ndarray, le) -> list:
    """Convert model indices to internal labels."""
    return [le.classes_[int(i)] for i in indices]


def map_internal_to_api(internal_label: int) -> int:
    """Map internal label (2,4,6) to API label (0,1,2)."""
    return INTERNAL_TO_API_LABELS.get(internal_label, internal_label)


def map_api_to_internal(api_label: int) -> int:
    """Map API label (0,1,2) to internal label (2,4,6)."""
    return API_TO_INTERNAL_LABELS.get(api_label, api_label)


def map_probabilities_to_api(proba: np.ndarray, le) -> dict[str, float]:
    """
    Map probability array to API labels (0,1,2).
    Input: proba array with internal label ordering
    Output: dict with API label keys (0,1,2)
    """
    internal_labels = le.classes_
    api_probs = {}
    for i, internal_label in enumerate(internal_labels):
        api_label = map_internal_to_api(int(internal_label))
        api_probs[str(api_label)] = float(proba[0, i])
    return api_probs
