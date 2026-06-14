from __future__ import annotations

import numpy as np

from src.preprocessing import encode_labels, normalize_pixels, stratified_splits, undersample_class


def test_normalize_and_splits():
    X = np.random.rand(100, 10).astype(np.float32) * 255.0
    Xn = normalize_pixels(X, 255.0)
    assert Xn.max() <= 1.0

    y = np.array([0] * 50 + [1] * 50)
    y2, le = encode_labels(y)
    assert set(np.unique(y2)) == {0, 1}

    a, b, c, ya, yb, yc = stratified_splits(Xn, y2, test_size=0.2, val_size=0.2, random_state=0)
    assert len(a) + len(b) + len(c) == 100


def test_undersample_class_nv_only():
    rng = np.random.default_rng(0)
    X = rng.random((120, 4)).astype(np.float32)
    y = np.array([0] * 80 + [1] * 25 + [2] * 15, dtype=np.int32)

    X_out, y_out = undersample_class(
        X, y, target_class=0, max_samples=30, random_state=42
    )

    assert len(y_out) == 30 + 25 + 15
    assert int(np.sum(y_out == 0)) == 30
    assert int(np.sum(y_out == 1)) == 25
    assert int(np.sum(y_out == 2)) == 15


def test_undersample_class_reproducible():
    X = np.arange(50, dtype=np.float32).reshape(50, 1)
    y = np.array([0] * 40 + [1] * 10, dtype=np.int32)

    X_a, y_a = undersample_class(X, y, target_class=0, max_samples=10, random_state=42)
    X_b, y_b = undersample_class(X, y, target_class=0, max_samples=10, random_state=42)

    np.testing.assert_array_equal(y_a, y_b)
    np.testing.assert_array_equal(X_a, X_b)


def test_undersample_class_no_op_when_below_cap():
    X = np.ones((20, 2), dtype=np.float32)
    y = np.array([0] * 5 + [1] * 15, dtype=np.int32)

    X_out, y_out = undersample_class(X, y, target_class=0, max_samples=3000, random_state=42)

    assert len(y_out) == 20
    np.testing.assert_array_equal(y, y_out)
