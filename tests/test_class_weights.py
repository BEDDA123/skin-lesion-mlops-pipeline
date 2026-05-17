from __future__ import annotations

import numpy as np

from src.class_weights import compute_balanced_class_weights, sample_weights_from_class_dict


def test_balanced_weights_boost_rare():
    y = np.array([4] * 100 + [3] * 5 + [5] * 5, dtype=np.int32)
    w = compute_balanced_class_weights(y, rare_classes=[3, 5], rare_class_boost=2.0)
    assert w[3] > w[4]
    assert w[5] > w[4]
    sw = sample_weights_from_class_dict(y, w)
    assert len(sw) == len(y)
