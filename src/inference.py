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

# Les labels sont maintenant directement (0,1,2) après le remappage
# 0 = nv (original 4), 1 = bkl (original 2), 2 = mel+bcc (original 6,1)
# Aucun remappage supplémentaire n'est nécessaire


def _artifacts_dir() -> Path:
    return ROOT / "models" / "artifacts"


def _processed_dir() -> Path:
    params = load_params()
    return ROOT / params["data"]["processed_dir"]


def load_label_encoder():
    return joblib.load(_processed_dir() / "label_encoder.joblib")


def load_scaler():
    """Charge le scaler si disponible."""
    scaler_path = _processed_dir() / "scaler.joblib"
    if scaler_path.is_file():
        return joblib.load(scaler_path)
    return None


def load_best_model() -> tuple[Any, Any]:
    """Charge le meilleur modèle (nom issu de best_model.json) et le LabelEncoder."""
    art = _artifacts_dir()
    with open(art / "best_model.json", encoding="utf-8") as f:
        best = json.load(f)
    name = str(best["model_name"])
    le = load_label_encoder()

    if name == "svm":
        return joblib.load(art / "svm.joblib"), le
    if name == "mlp":
        return joblib.load(art / "mlp.joblib"), le
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
    # Appliquer le scaler si disponible
    scaler = load_scaler()
    if scaler is not None:
        X_flat = scaler.transform(X_flat)

    proba = model.predict_proba(X_flat)
    pred_idx = np.argmax(proba, axis=1)
    return pred_idx, proba


def indices_to_labels(indices: np.ndarray, le) -> list:
    """Convert model indices to internal labels."""
    return [le.classes_[int(i)] for i in indices]


def map_probabilities_to_api(proba: np.ndarray, le) -> dict[str, float]:
    """
    Convertir le tableau de probabilités en dict avec clés de labels.
    Puisque les labels sont déjà (0,1,2), pas de remappage supplémentaire.
    
    Args:
        proba: array de probabilités (shape: batch_size x n_classes)
        le: LabelEncoder avec classes_ = [0, 1, 2]
    
    Returns:
        dict: {"0": prob0, "1": prob1, "2": prob2}
    """
    class_labels = le.classes_
    probs_dict = {}
    for i, label in enumerate(class_labels):
        probs_dict[str(int(label))] = float(proba[0, i])
    return probs_dict
