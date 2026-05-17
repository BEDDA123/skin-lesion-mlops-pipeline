from __future__ import annotations

import logging

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


def compute_balanced_class_weights(
    y: np.ndarray,
    *,
    rare_classes: list[int] | None = None,
    rare_class_boost: float = 1.0,
) -> dict[int, float]:
    """
    Poids balanced sklearn, avec renforcement optionnel des classes rares.
    rare_classes : indices de classes encodés (ex. 3 et 5 pour les lésions les plus rares).
    
    Formule :
    - w_c = N / (K * n_c)  où N = total samples, K = n_classes, n_c = count of class c
    - Si c ∈ rare_classes : w_c *= rare_class_boost
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    w_map = {int(c): float(w) for c, w in zip(classes, weights)}

    if rare_classes and rare_class_boost != 1.0:
        for cls in rare_classes:
            idx = int(cls)
            if idx in w_map:
                w_map[idx] *= float(rare_class_boost)
                logger.info(f"Boost applied to class {idx}: {w_map[idx]:.4f}")

    return w_map


def compute_aggressive_class_weights(
    y: np.ndarray,
    *,
    rare_classes: list[int] | None = None,
    strategy: str = "inverse_frequency",
) -> dict[int, float]:
    """
    Stratégie plus agressive pour les classes rares.
    
    Stratégies :
    - 'balanced': sklearn standard (recommandé pour la plupart)
    - 'inverse_frequency': w_c = 1 / n_c (pas de normalisation)
    - 'logarithmic': w_c = log(N / n_c) (moins agressif)
    - 'sqrt': w_c = sqrt(N / n_c) (compromis)
    """
    classes = np.unique(y)
    unique, counts = np.unique(y, return_counts=True)
    class_count = dict(zip(unique, counts))
    N = len(y)
    K = len(classes)
    
    w_map = {}
    
    if strategy == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=y)
        w_map = {int(c): float(w) for c, w in zip(classes, weights)}
    
    elif strategy == "inverse_frequency":
        for cls in classes:
            w_map[int(cls)] = 1.0 / class_count[cls]
    
    elif strategy == "logarithmic":
        for cls in classes:
            w_map[int(cls)] = np.log(N / class_count[cls])
    
    elif strategy == "sqrt":
        for cls in classes:
            w_map[int(cls)] = np.sqrt(N / class_count[cls])
    
    else:
        raise ValueError(f"Stratégie inconnue: {strategy}")
    
    # Normalisation optionnelle : moyenne = 1
    mean_weight = np.mean(list(w_map.values()))
    w_map = {cls: w / mean_weight for cls, w in w_map.items()}
    
    # Boost pour classes rares
    if rare_classes:
        for cls in rare_classes:
            if int(cls) in w_map:
                w_map[int(cls)] *= 2.0  # agressif pour les très rares
    
    logger.info(f"Weights ({strategy}): {w_map}")
    return w_map


def compute_entropy_based_weights(y: np.ndarray) -> dict[int, float]:
    """
    Poids basés sur l'entropie Shannon de la distribution des classes.
    Classes rares ont naturellement plus de poids.
    
    w_c = -log(P_c) où P_c = n_c / N
    """
    unique, counts = np.unique(y, return_counts=True)
    N = len(y)
    
    w_map = {}
    for cls, count in zip(unique, counts):
        p_c = count / N
        # Éviter log(0)
        w_c = -np.log(np.clip(p_c, 1e-10, 1.0))
        w_map[int(cls)] = float(w_c)
    
    # Normalisation
    mean_weight = np.mean(list(w_map.values()))
    w_map = {cls: w / mean_weight for cls, w in w_map.items()}
    
    logger.info(f"Entropy-based weights: {w_map}")
    return w_map


def sample_weights_from_class_dict(y: np.ndarray, class_weights: dict[int, float]) -> np.ndarray:
    """Convertit un dict {class: weight} en array de sample weights."""
    return np.array([class_weights[int(label)] for label in y], dtype=np.float64)


def analyze_class_weights(class_weights: dict[int, float]) -> dict:
    """Analyse et affiche les propriétés des poids de classe."""
    weights_array = np.array(list(class_weights.values()))
    return {
        "min_weight": float(np.min(weights_array)),
        "max_weight": float(np.max(weights_array)),
        "mean_weight": float(np.mean(weights_array)),
        "std_weight": float(np.std(weights_array)),
        "weight_ratio": float(np.max(weights_array) / np.min(weights_array)),
    }
