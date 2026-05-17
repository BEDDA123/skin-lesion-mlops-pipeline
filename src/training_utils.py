from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def maybe_subsample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not sample_rows or len(X_train) <= sample_rows:
        return X_train, y_train, X_val, y_val
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=sample_rows, replace=False)
    idx_val = rng.choice(len(X_val), size=min(sample_rows // 4, len(X_val)), replace=False)
    return X_train[idx], y_train[idx], X_val[idx_val], y_val[idx_val]
