from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend to prevent tkinter errors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def _to_json_serializable(obj):
    """Convertit récursivement numpy / clés non-str pour json.dump."""
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def classification_report_dict(y_true, y_pred, label_encoder: LabelEncoder | None = None) -> dict:
    names = None
    if label_encoder is not None:
        names = [str(c) for c in label_encoder.classes_]
    rep = classification_report(
        y_true, y_pred, target_names=names, output_dict=True, zero_division=0
    )
    return _to_json_serializable(rep)


def save_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    label_encoder: LabelEncoder | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = None
    if label_encoder is not None:
        labels = list(label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, ax=ax, colorbar=False
    )
    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Matrice de confusion enregistrée: %s", out_path)
