from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def normalize_pixels(X: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    return np.clip(X, 0, None) / float(max_val)


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode labels and remap to 0, 1, 2 for 3-class classification.
    Original classes (2, 4, 6) are remapped to (0, 1, 2).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Remap to sequential 0, 1, 2 if we have 3 classes
    unique_classes = sorted(np.unique(y_enc))
    if len(unique_classes) == 3:
        # Create mapping from original encoded to sequential 0, 1, 2
        mapping = {old: new for new, old in enumerate(unique_classes)}
        y_enc = np.array([mapping[label] for label in y_enc], dtype=np.int32)

        # Update LabelEncoder classes to reflect remapped labels
        le.classes_ = np.array([0, 1, 2], dtype=int)
        logger.info(f"Labels remapped: {unique_classes} -> [0, 1, 2]")
        logger.info(f"LabelEncoder classes updated to: {le.classes_}")

    return y_enc, le


def stratified_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train / validation / test stratifiés."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_imbalance_method(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    No resampling - returns original training data.
    All imbalance handling removed to use real data distribution.
    """
    logger.info("Using original training data without resampling")
    return X_train, y_train


def save_processed(
    out_dir: Path,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    label_encoder: LabelEncoder,
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dataset.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    joblib.dump(label_encoder, out_dir / "label_encoder.joblib")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Données preprocessées sauvegardées dans %s", out_dir)


def load_processed(processed_dir: Path) -> dict:
    data = np.load(processed_dir / "dataset.npz")
    le: LabelEncoder = joblib.load(processed_dir / "label_encoder.joblib")
    with open(processed_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "y_test": data["y_test"],
        "label_encoder": le,
        "meta": meta,
    }
