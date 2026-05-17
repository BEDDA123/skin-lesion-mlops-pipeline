from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV (une ligne = une image, dernière colonne = label)."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable: {csv_path}. "
            "Copiez le CSV dans data/raw/ ou définissez SKIN_CSV_PATH."
        )
    df = pd.read_csv(csv_path)
    return df


def validate_dataset(df: pd.DataFrame, label_col: str = "label") -> dict:
    """Vérifie les NaN, la cohérence des colonnes et retourne un résumé."""
    if label_col not in df.columns:
        raise ValueError(f"Colonne label '{label_col}' absente. Colonnes: {list(df.columns[:5])}…")

    feature_cols = [c for c in df.columns if c != label_col]
    missing = df.isnull().sum()
    n_missing_total = int(missing.sum())

    summary = {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "label_col": label_col,
        "n_missing_values": n_missing_total,
        "missing_per_column_top": missing[missing > 0]
        .sort_values(ascending=False)
        .head(10)
        .to_dict(),
        "label_counts": df[label_col].value_counts().sort_index().to_dict(),
        "dtypes": {label_col: str(df[label_col].dtype), "pixels": "numeric"},
    }

    if n_missing_total:
        logger.warning("Valeurs manquantes détectées: %s", n_missing_total)
    else:
        logger.info("Aucune valeur manquante.")

    return summary


def split_features_labels(df: pd.DataFrame, label_col: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """Sépare X (float32) et y (int ou str selon CSV)."""
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    return X, y


def log_dataset_stats(summary: dict) -> None:
    logger.info("Statistiques dataset: %s lignes, %s features", summary["n_rows"], summary["n_features"])
    logger.info("Distribution des labels: %s", summary["label_counts"])
