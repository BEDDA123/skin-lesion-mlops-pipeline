from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion import split_features_labels, validate_dataset


def test_validate_and_split(tmp_path: Path):
    n_pix = 28 * 28 * 3
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, size=(30, n_pix))
    y = rng.integers(0, 3, size=30)
    cols = [f"pixel{i:04d}" for i in range(n_pix)] + ["label"]
    df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=cols)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)

    d = pd.read_csv(p)
    s = validate_dataset(d)
    assert s["n_rows"] == 30
    assert s["n_features"] == n_pix

    Xf, yv = split_features_labels(d)
    assert Xf.shape == (30, n_pix)
    assert len(yv) == 30
