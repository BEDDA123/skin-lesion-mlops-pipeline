from __future__ import annotations

import numpy as np

from src.preprocessing import encode_labels, normalize_pixels, stratified_splits


def test_normalize_and_splits():
    X = np.random.rand(100, 10).astype(np.float32) * 255.0
    Xn = normalize_pixels(X, 255.0)
    assert Xn.max() <= 1.0

    y = np.array([0] * 50 + [1] * 50)
    y2, le = encode_labels(y)
    assert set(np.unique(y2)) == {0, 1}

    a, b, c, ya, yb, yc = stratified_splits(Xn, y2, test_size=0.2, val_size=0.2, random_state=0)
    assert len(a) + len(b) + len(c) == 100
