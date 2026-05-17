"""
Diagnostic approfondi des problèmes de classification.
Analyse les probabilités, la distribution des classes et identifie les goulots d'étranglement.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import (
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def analyze_predictions_entropy(
    y_true: np.ndarray, y_pred_proba: np.ndarray, label_encoder=None
) -> dict:
    """
    Analyse l'entropie des probabilités prédites.
    Entropie faible = confiance élevée (bon).
    Entropie proche de log(n_classes) = doute (mauvais).
    """
    n_classes = y_pred_proba.shape[1]
    log_nc = np.log(n_classes)

    # Entropie par prédiction
    entropies = entropy(y_pred_proba.T)

    # Confiance (max proba)
    max_probas = np.max(y_pred_proba, axis=1)

    return {
        "entropy_mean": float(np.mean(entropies)),
        "entropy_median": float(np.median(entropies)),
        "entropy_std": float(np.std(entropies)),
        "entropy_max": float(np.max(entropies)),
        "max_proba_mean": float(np.mean(max_probas)),
        "max_proba_min": float(np.min(max_probas)),
        "max_proba_median": float(np.median(max_probas)),
        "log_n_classes": float(log_nc),
        "entropy_ratio_mean": float(np.mean(entropies) / log_nc),  # 0 = certain, 1 = très doute
        "n_high_entropy_predictions": int(
            np.sum(entropies > log_nc * 0.8)
        ),  # prédictions hésitantes
    }


def analyze_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
    label_encoder=None,
) -> dict:
    """
    Analyse les métriques par classe : recall, precision, f1, entropie moyenne.
    """
    from sklearn.metrics import precision_recall_fscore_support

    classes = np.unique(y_true)
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class = {}
    for cls, p, r, f, sup in zip(classes, precision, recall, fscore, support):
        class_label = str(label_encoder.classes_[int(cls)]) if label_encoder else str(cls)
        class_mask = y_true == cls

        metrics = {
            "support": int(sup),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }

        if y_pred_proba is not None:
            # Confiance moyenne pour cette classe
            class_entropies = entropy(y_pred_proba[class_mask].T)
            metrics["entropy_mean"] = float(np.mean(class_entropies))
            metrics["max_proba_mean"] = float(np.mean(np.max(y_pred_proba[class_mask], axis=1)))

        per_class[class_label] = metrics

    return per_class


def analyze_confusion_imbalance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Analyse la matrice de confusion pour identifier les patterns de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)

    # Taux de confusion : combien de fois classe i est classée en j
    confusion_pairs = {}
    for i, cls_i in enumerate(classes):
        row = cm[i]
        total = row.sum()
        if total > 0:
            # Les plus grandes sources de confusion
            top_confusion_idx = np.argsort(row)[-3:][::-1]
            for idx in top_confusion_idx[1:]:  # skip le premier (c'est la diagonale)
                if row[idx] > 0:
                    cls_j = classes[idx]
                    confusion_pairs[f"{cls_i}->{cls_j}"] = {
                        "count": int(row[idx]),
                        "percentage": float(100 * row[idx] / total),
                    }

    return {"top_confusions": confusion_pairs}


def analyze_class_distribution(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Analyse la distribution des classes à travers train/val/test.
    """

    def dist_dict(y):
        unique, counts = np.unique(y, return_counts=True)
        return {str(int(c)): int(cnt) for c, cnt in zip(unique, counts)}

    dist_train = dist_dict(y_train)
    dist_val = dist_dict(y_val)
    dist_test = dist_dict(y_test)

    # Imbalance ratios
    all_counts_train = list(dist_train.values())
    imbalance_ratio = max(all_counts_train) / min(all_counts_train) if all_counts_train else 0

    return {
        "train_distribution": dist_train,
        "val_distribution": dist_val,
        "test_distribution": dist_test,
        "imbalance_ratio": float(imbalance_ratio),
        "n_train": int(np.sum(list(dist_train.values()))),
        "n_val": int(np.sum(list(dist_val.values()))),
        "n_test": int(np.sum(list(dist_test.values()))),
    }


def analyze_feature_importance(
    model, feature_names: list[str] | None = None, top_n: int = 20
) -> dict:
    """
    Extrait l'importance des features pour XGBoost et RandomForest.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        top_features = {}
        for rank, idx in enumerate(indices, 1):
            top_features[
                f"{rank}. {feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'}"
            ] = float(importances[idx])

        return {"top_features": top_features, "total_features": len(importances)}
    elif hasattr(model, "coef_"):
        # Logistic Regression
        coef_abs = np.abs(model.coef_).mean(axis=0)
        indices = np.argsort(coef_abs)[::-1][:top_n]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coef_abs))]

        top_features = {}
        for rank, idx in enumerate(indices, 1):
            top_features[
                f"{rank}. {feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'}"
            ] = float(coef_abs[idx])

        return {"top_features": top_features, "total_features": len(coef_abs)}

    return {"note": "Model type not supported for feature importance"}


def run_full_diagnostic(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    label_encoder=None,
    output_path: Path | None = None,
) -> dict:
    """
    Lance un diagnostic complet sur validation et test.
    """
    logger.info("Démarrage du diagnostic complet...")

    # Prédictions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    y_val_proba = None
    y_test_proba = None
    if hasattr(model, "predict_proba"):
        y_val_proba = model.predict_proba(X_val)
        y_test_proba = model.predict_proba(X_test)

    diagnostic = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_distribution": analyze_class_distribution(y_train, y_val, y_test),
        "validation_entropy": analyze_predictions_entropy(y_val, y_val_proba, label_encoder)
        if y_val_proba is not None
        else {},
        "test_entropy": analyze_predictions_entropy(y_test, y_test_proba, label_encoder)
        if y_test_proba is not None
        else {},
        "val_per_class_metrics": analyze_per_class_metrics(
            y_val, y_val_pred, y_val_proba, label_encoder
        ),
        "test_per_class_metrics": analyze_per_class_metrics(
            y_test, y_test_pred, y_test_proba, label_encoder
        ),
        "test_confusion_analysis": analyze_confusion_imbalance(y_test, y_test_pred),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(diagnostic, f, indent=2)
        logger.info("Diagnostic sauvegardé dans %s", output_path)

    return diagnostic


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Diagnostic module ready. Use run_full_diagnostic() to analyze model performance.")
