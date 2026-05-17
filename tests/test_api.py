from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np

from src.api.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body


def test_predict_when_model_present():
    # Mock the model loading to avoid requiring actual training
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.7, 0.2, 0.1]])
    mock_model.predict.return_value = np.array([0])

    mock_encoder = MagicMock()
    mock_encoder.classes_ = np.array(["0", "1", "2"])

    # Patch the global variables directly instead of triggering startup
    with patch("src.api.main._model", mock_model), \
         patch("src.api.main._label_encoder", mock_encoder):
        client = TestClient(app)
        vec = [float(x) for x in range(2352)]
        r = client.post("/predict", json={"pixels": vec, "normalize": False})
        assert r.status_code == 200
        data = r.json()
        assert "predicted_label" in data
