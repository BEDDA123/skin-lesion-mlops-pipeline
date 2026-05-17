"""
Modules avancés pour calibration probabiliste et tuning d'hyperparamètres.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Calibration simple via temperature scaling.
    Divise les logits par une température T > 1 pour lisser les probabilités.

    Usage:
        ts = TemperatureScaling()
        ts.fit(model, X_val, y_val)
        y_proba_calibrated = ts.transform(model.predict_proba(X_test))
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Trouve la température T qui minimise la NLL sur validation.
        """
        y_proba = model.predict_proba(X_val)

        best_temp = 1.0
        best_loss = float("inf")

        for T in np.linspace(0.5, 5.0, 50):
            y_proba_scaled = self._scale_proba(y_proba, T)
            loss = log_loss(y_val, y_proba_scaled)
            if loss < best_loss:
                best_loss = loss
                best_temp = T

        self.temperature = best_temp
        logger.info(f"Temperature scaling fitted: T={self.temperature:.4f}, loss={best_loss:.4f}")

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        """Applique le temperature scaling."""
        return self._scale_proba(y_proba, self.temperature)

    @staticmethod
    def _scale_proba(y_proba: np.ndarray, temperature: float) -> np.ndarray:
        """Divise par T et renormalise."""
        scaled = np.power(y_proba, 1.0 / temperature)
        return scaled / np.sum(scaled, axis=1, keepdims=True)


def apply_calibration(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """
    Applique la calibration probabiliste via sklearn.

    Méthodes :
    - 'sigmoid': régression logistique (recommandé)
    - 'isotonic': plus flexible mais nécessite plus de données
    """
    logger.info(f"Calibrating model with method={method}")

    # Use cross-validation on validation set (cv="prefit" removed in sklearn 1.5+)
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv=5,
    )

    calibrated.fit(X_val, y_val)
    logger.info("Model calibration complete")

    return calibrated


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_metric: str = "f1_macro",
    class_weights: dict[int, float] | None = None,
) -> dict[str, Any]:
    """
    Trouve le seuil de décision optimal pour maximiser une métrique.

    Pour multi-classe, on peut ajuster les seuils par classe ou utiliser
    une stratégie de confiance minimale.

    Retourne :
    - optimal_threshold: seuil de confiance
    - metric_values: évolution de la métrique
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.linspace(0.0, 1.0, 101)
    metric_values = []

    for threshold in thresholds:
        # Pour chaque exemple, prendre la classe si confidence > threshold
        y_pred = np.argmax(y_proba, axis=1)
        max_proba = np.max(y_proba, axis=1)

        # Conserver seulement les prédictions confiantes
        confident_mask = max_proba >= threshold

        if confident_mask.sum() == 0:
            metric_values.append(0.0)
            continue

        if target_metric == "f1_macro":
            metric = f1_score(
                y_true[confident_mask],
                y_pred[confident_mask],
                average="macro",
                zero_division=0,
            )
        elif target_metric == "recall_macro":
            metric = recall_score(
                y_true[confident_mask],
                y_pred[confident_mask],
                average="macro",
                zero_division=0,
            )
        else:
            metric = precision_score(
                y_true[confident_mask],
                y_pred[confident_mask],
                average="macro",
                zero_division=0,
            )

        metric_values.append(metric)

    optimal_idx = np.argmax(metric_values)
    optimal_threshold = thresholds[optimal_idx]

    logger.info(f"Optimal threshold for {target_metric}: {optimal_threshold:.4f}")

    return {
        "optimal_threshold": float(optimal_threshold),
        "optimal_metric_value": float(metric_values[optimal_idx]),
        "thresholds": thresholds.tolist(),
        "metric_values": metric_values,
    }


def estimate_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Estime l'erreur d'étalonnage via Expected Calibration Error (ECE).

    ECE = moyenne pondérée des |confiance - accuracy| par bin
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = []
    confidences = []
    bin_sizes = []

    max_proba = np.max(y_proba, axis=1)
    y_pred = np.argmax(y_proba, axis=1)
    correct = (y_pred == y_true).astype(float)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_proba >= bin_lower) & (max_proba < bin_upper)
        bin_size = in_bin.sum()
        bin_sizes.append(int(bin_size))

        if bin_size > 0:
            accuracy_in_bin = correct[in_bin].mean()
            confidence_in_bin = max_proba[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(confidence_in_bin)
        else:
            accuracies.append(0.0)
            confidences.append(0.0)

    # ECE : moyenne pondérée
    ece = np.average(
        np.abs(np.array(accuracies) - np.array(confidences)),
        weights=np.array(bin_sizes),
    )

    return {
        "ece": float(ece),
        "bin_accuracies": accuracies,
        "bin_confidences": confidences,
        "bin_sizes": bin_sizes,
        "interpretation": "ECE proche de 0 = bien calibré, ECE > 0.1 = mal calibré",
    }
