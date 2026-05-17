from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from src.config import ROOT, load_params

logger = logging.getLogger(__name__)


def _log_dir() -> Path:
    p = ROOT / load_params()["monitoring"]["log_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mean_abs_drift(reference: np.ndarray, batch: np.ndarray, max_dims: int = 32) -> float:
    """Moyenne des différences |mean_ref - mean_batch| sur les premières dimensions (proxy léger)."""
    ref = reference[:, :max_dims].mean(axis=0)
    bat = batch[:, :max_dims].mean(axis=0)
    return float(np.mean(np.abs(ref - bat)))


def _ks_drift(reference: np.ndarray, batch: np.ndarray, dim: int = 0) -> dict:
    """Test KS sur une dimension de pixels (indicateur de dérive marges)."""
    stat, pvalue = ks_2samp(reference[:, dim], batch[:, dim])
    return {"ks_stat_dim0": float(stat), "ks_pvalue_dim0": float(pvalue)}


def log_prediction_batch(
    y_pred: np.ndarray,
    proba: np.ndarray,
    latency_ms: float,
    reference_X: np.ndarray | None,
    X_batch: np.ndarray,
) -> None:
    """Append JSONL: latence, confiance max, proxy drift."""
    logf = _log_dir() / f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency_ms,
        "max_proba": float(np.max(proba)),
        "pred": int(y_pred[0]),
    }
    if reference_X is not None and X_batch.size:
        try:
            record["mean_abs_drift"] = _mean_abs_drift(reference_X, X_batch)
            record.update(_ks_drift(reference_X, X_batch))
        except Exception as e:  # noqa: BLE001
            record["drift_error"] = str(e)
    with open(logf, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Prédiction journalisée: %s", record)
