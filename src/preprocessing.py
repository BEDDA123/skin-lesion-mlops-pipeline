from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTETomek

logger = logging.getLogger(__name__)


def normalize_pixels(X: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    """Normalisation simple (division par 255)."""
    return np.clip(X, 0, None) / float(max_val)


def normalize_pixels_advanced(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = "standard"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalisation avancée avec fit sur train uniquement pour éviter data leakage.
    
    Args:
        X_train, X_val, X_test: Données à normaliser
        method: 'standard', 'minmax', 'robust'
    
    Returns:
        X_train_norm, X_val_norm, X_test_norm, scaler
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Method {method} not supported")
    
    # Fit sur train uniquement
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    logger.info(f"Normalization applied: {method}")
    logger.info(f"Train mean: {X_train_norm.mean():.4f}, std: {X_train_norm.std():.4f}")
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


def apply_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Applique PCA avec conservation de variance spécifiée.
    
    Args:
        X_train, X_val, X_test: Données normalisées
        variance_threshold: Fraction de variance à conserver (0.95 = 95%)
        max_components: Nombre maximum de components
    
    Returns:
        X_train_pca, X_val_pca, X_test_pca, pca_model
    """
    # Fit PCA sur train uniquement
    pca = PCA(n_components=min(max_components, X_train.shape[1]))
    pca.fit(X_train)
    
    # Trouver le nombre de components pour variance_threshold
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    logger.info(f"PCA: {n_components} components pour {variance_threshold*100:.1f}% variance")
    logger.info(f"Variance expliquée: {cumulative_variance[n_components-1]:.4f}")
    
    # Refit avec le bon nombre de components
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    return X_train_pca, X_val_pca, X_test_pca, pca


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode et valide les labels pour la classification 3-classes.
    
    Attendu: les labels sont déjà (0, 1, 2) après le prétraitement dans data.py
    - 0 = nv (original label 4)
    - 1 = bkl (original label 2)
    - 2 = mel+bcc (original labels 1, 6)
    
    Les classes supprimées (0, 3, 5) ont été supprimées du dataset.
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
    method: str = "none",
    sampling_strategy: str | dict = "auto"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique une méthode de gestion du déséquilibre.
    
    Args:
        X_train, y_train: Données d'entraînement
        method: 'none', 'smote', 'smote_tomek', 'undersample'
        sampling_strategy: 'auto' ou dict {class: ratio}
    
    Returns:
        X_train_bal, y_train_bal
    """
    if method == "none":
        logger.info("Using original training data without resampling")
        return X_train, y_train
    
    elif method == "smote":
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE applied: {len(X_train)} -> {len(X_train_bal)} samples")
        return X_train_bal, y_train_bal
    
    elif method == "smote_tomek":
        smote_tomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
        X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)
        logger.info(f"SMOTE+Tomek applied: {len(X_train)} -> {len(X_train_bal)} samples")
        logger.info(f"Distribution après SMOTE+Tomek: {np.bincount(y_train_bal)}")
        return X_train_bal, y_train_bal
    
    elif method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_train_bal, y_train_bal = undersampler.fit_resample(X_train, y_train)
        logger.info(f"Undersampling applied: {len(X_train)} -> {len(X_train_bal)} samples")
        return X_train_bal, y_train_bal
    
    else:
        raise ValueError(f"Unknown imbalance method: {method}")


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
