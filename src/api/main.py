from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ROOT, load_params
from src.inference import (
    indices_to_labels,
    load_best_model,
    map_internal_to_api,
    map_probabilities_to_api,
    predict_batch,
)
from src.monitoring.metrics import log_prediction_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

_model = None
_label_encoder = None
_reference_X: np.ndarray | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - replaces deprecated @app.on_event."""
    global _model, _label_encoder, _reference_X

    # Startup
    try:
        _model, _label_encoder = load_best_model()
        logger.info("Modèle chargé: classes=%s", list(_label_encoder.classes_))
    except Exception as e:  # noqa: BLE001 — démarrage sans artefacts (tests / CI)
        logger.warning("Chargement modèle impossible: %s", e)
        _model, _label_encoder = None, None

    try:
        data = np.load(ROOT / load_params()["data"]["processed_dir"] / "dataset.npz")
        ref_n = int(load_params()["monitoring"]["reference_sample_size"])
        Xr = data["X_train"][:ref_n]
        _reference_X = np.asarray(Xr, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        logger.warning("Référence drift non chargée: %s", e)
        _reference_X = None

    yield

    # Shutdown (cleanup if needed)
    logger.info("API shutdown")


app = FastAPI(
    title="API classification lésions cutanées",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Vecteur aplati des pixels (valeurs 0–255 ou déjà normalisées 0–1)."""

    pixels: list[float] = Field(..., description="Liste de longueur n_pixels (ex. 2352 pour 28x28x3)")
    normalize: bool = True


class PredictResponse(BaseModel):
    predicted_index: int
    predicted_label: str
    probabilities: dict[str, float]
    latency_ms: float


@app.get("/health")
def health():
    ok = _model is not None
    return {"status": "ok" if ok else "degraded", "model_loaded": ok}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(503, "Modèle non chargé. Lancez l'entraînement (scripts ou DVC).")

    t0 = time.perf_counter()
    arr = np.array(req.pixels, dtype=np.float32).reshape(1, -1)
    if req.normalize:
        arr = np.clip(arr, 0, None) / 255.0

    pred_idx, proba = predict_batch(_model, arr, _label_encoder)
    labels = _label_encoder.classes_
    probs = {str(labels[i]): float(proba[0, i]) for i in range(len(labels))}
    latency = (time.perf_counter() - t0) * 1000

    log_prediction_batch(
        y_pred=pred_idx,
        proba=proba,
        latency_ms=latency,
        reference_X=_reference_X,
        X_batch=arr,
    )

    return PredictResponse(
        predicted_index=int(pred_idx[0]),
        predicted_label=str(indices_to_labels(pred_idx, _label_encoder)[0]),
        probabilities=probs,
        latency_ms=float(latency),
    )


@app.get("/metrics/summary")
def metrics_summary():
    """Lecture simple du dernier fichier de métriques batch (démo monitoring)."""
    log_dir = ROOT / load_params()["monitoring"]["log_dir"]
    files = sorted(log_dir.glob("predictions_*.jsonl"))
    if not files:
        return {"message": "Aucun log encore."}
    lines = files[-1].read_text(encoding="utf-8").strip().splitlines()
    return {"file": str(files[-1]), "n_lines": len(lines), "last": json.loads(lines[-1]) if lines else None}
