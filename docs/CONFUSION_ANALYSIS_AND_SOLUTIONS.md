# Analyse de la Confusion et Solutions pour la Classification du Cancer de la Peau

## Table des Matières

1. [Analyse du Problème de Confusion](#analyse-du-problème-de-confusion)
2. [Causes Possibles de la Confusion](#causes-possibles-de-la-confusion)
3. [Solutions pour la Normalisation des Données](#solutions-pour-la-normalisation-des-données)
4. [PCA et Réduction de Dimension](#pca-et-réduction-de-dimension)
5. [Gestion du Déséquilibre des Classes](#gestion-du-déséquilibre-des-classes)
6. [Optimisation des Hyperparamètres](#optimisation-des-hyperparamètres)
7. [Sélection de Features](#sélection-de-features)
8. [Amélioration des Métriques (Precision, Recall, F1)](#amélioration-des-métriques)
9. [Réduction de l'Overfitting](#réduction-de-loverfitting)
10. [Amélioration de la Matrice de Confusion](#amélioration-de-la-matrice-de-confusion)
11. [Exemples de Code Python](#exemples-de-code-python)
12. [Comparaison des Modèles](#comparaison-des-modèles)
13. [Limites des Modèles Classiques](#limites-des-modèles-classiques)
14. [Bonnes Pratiques MLOps](#bonnes-pratiques-mlops)

---

## Analyse du Problème de Confusion

### Contexte Actuel

Votre projet utilise des données CSV contenant des pixels d'images médicales pour classifier 3 types de lésions cutanées:

- **Classe 0 (nv)**: Melanocytic nevi - 1099 samples (12.4%)
- **Classe 1 (mel)**: Melanoma - 6705 samples (75.6%) - MAJORITAIRE
- **Classe 2 (bcc)**: Basal cell carcinoma - 1113 samples (12.6%)

### Pourquoi la Confusion Apparaît

#### 1. **Similarité Visuelle des Lésions**

Les lésions cutanées ont souvent des caractéristiques visuelles similaires:
- **Nevi (naevus)** vs **Melanome**: Les deux peuvent avoir des bordures irrégulières, des variations de couleur
- **Melanome** vs **BCC**: Partagent des caractéristiques comme des ulcérations, des saignements
- **Nevi** vs **BCC**: Peuvent tous deux être des lésions pigmentées

#### 2. **Limitations des Données CSV de Pixels**

Les données CSV de pixels (probablement 28x28x3 = 2352 features) présentent des problèmes:
- **Perte de contexte spatial**: Les modèles classiques ne capturent pas les relations spatiales entre pixels
- **Flattening**: L'image 2D/3D est aplatie en 1D, détruisant la structure locale
- **Haute dimensionnalité**: 2352 features pour seulement 8917 samples → curse of dimensionality

#### 3. **Déséquilibre des Classes**

- Classe majoritaire (mel) représente 75.6% des données
- Classes minoritaires (nv, bcc) seulement 12.4% et 12.6%
- Le modèle peut biais vers la prédiction de la classe majoritaire

#### 4. **Qualité des Features**

Les pixels bruts contiennent beaucoup de bruit:
- Variation d'éclairage
- Artefacts d'acquisition
- Background non pertinent
- Redondance d'information (pixels corrélés)

---

## Causes Possibles de la Confusion

### 1. **Overfitting sur la Classe Majoritaire**

```python
# Symptôme: Le modèle prédit principalement la classe 1 (mel)
# Cause: Déséquilibre des classes non suffisamment corrigé
```

### 2. **Features Non Discriminantes**

```python
# Symptôme: Les features (pixels) ne séparent pas bien les classes
# Cause: Pixels bruts sans extraction de caractéristiques
```

### 3. **Haute Dimensionnalité**

```python
# Symptôme: Performance médiocre malgré beaucoup de features
# Cause: Curse of dimensionality - trop de features par rapport aux samples
```

### 4. **Manque de Régularisation**

```python
# Symptôme: Bonnes performances train, mauvaises performances test
# Cause: Modèle trop complexe pour la quantité de données
```

### 5. **Similarité Intra-classe Élevée**

```python
# Symptôme: Grande variance dans les prédictions pour une même classe
# Cause: Les lésions d'une même classe peuvent être très différentes visuellement
```

---

## Solutions pour la Normalisation des Données

### Normalisation Actuelle (Simple)

```python
def normalize_pixels(X: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    return np.clip(X, 0, None) / float(max_val)
```

**Problème**: Division par 255 seulement, pas de standardisation.

### Solutions Améliorées

#### 1. **Standardisation (Z-score)**

```python
from sklearn.preprocessing import StandardScaler

def advanced_normalization(X_train, X_val, X_test):
    """
    Standardisation avec fit sur train uniquement pour éviter data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
```

**Avantages**:
- Centre les données (moyenne = 0)
- Réduit l'échelle (écart-type = 1)
- Meilleure convergence pour Logistic Regression et XGBoost

#### 2. **Min-Max Scaling**

```python
from sklearn.preprocessing import MinMaxScaler

def minmax_normalization(X_train, X_val, X_test, feature_range=(0, 1)):
    """
    Min-Max scaling avec fit sur train uniquement.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
```

**Avantages**:
- Garde les valeurs dans une plage spécifique
- Utile pour les modèles sensibles à l'échelle

#### 3. **Robust Scaling (résistant aux outliers)**

```python
from sklearn.preprocessing import RobustScaler

def robust_normalization(X_train, X_val, X_test):
    """
    Robust scaling utilisant median et IQR au lieu de mean et std.
    Résistant aux outliers dans les pixels.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
```

**Avantages**:
- Résistant aux outliers (pixels extrêmes)
- Utilise median et IQR (interquartile range)

#### 4. **Normalisation par Canal (pour RGB)**

```python
def channel_normalization(X):
    """
    Normalise chaque canal RGB indépendamment.
    X shape: (n_samples, n_features) où n_features = height * width * channels
    """
    n_samples = X.shape[0]
    height = width = 28  # selon votre dataset
    channels = 3
    
    X_reshaped = X.reshape(n_samples, height, width, channels)
    
    X_normalized = np.zeros_like(X_reshaped)
    for c in range(channels):
        channel_data = X_reshaped[:, :, :, c]
        mean = channel_data.mean()
        std = channel_data.std()
        X_normalized[:, :, :, c] = (channel_data - mean) / (std + 1e-8)
    
    return X_normalized.reshape(n_samples, -1)
```

### Recommandation pour Votre Projet

```python
# Dans src/preprocessing.py, ajoutez:

from sklearn.preprocessing import StandardScaler
import joblib

def normalize_pixels_advanced(
    X_train: np.ndarray, 
    X_val: np.ndarray, 
    X_test: np.ndarray,
    method: str = "standard"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalisation avancée avec fit sur train uniquement.
    
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
```

---

## PCA et Réduction de Dimension

### Pourquoi Utiliser PCA?

1. **Réduction de la dimensionnalité**: 2352 features → ~100-500 components
2. **Décorrélation**: Les components sont orthogonaux
3. **Débruitage**: Les components de faible variance contiennent souvent du bruit
4. **Amélioration de la performance**: Moins de features = moins d'overfitting

### Implémentation PCA

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

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
    X_train_pca = pca.fit_transform(X_train)
    
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


def plot_pca_variance(pca: PCA, save_path: str = None):
    """
    Visualise la variance expliquée par PCA.
    """
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% variance')
    plt.xlabel('Nombre de Components')
    plt.ylabel('Variance Cumulative Expliquée')
    plt.title('PCA - Variance Expliquée')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Intégration dans Votre Pipeline

```python
# Dans src/preprocessing.py, ajoutez:

def apply_pca_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Pipeline complet: Normalisation + PCA
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # 1. Standardisation
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    
    # 2. PCA
    pca = PCA(n_components=min(max_components, X_train_std.shape[1]))
    pca.fit(X_train_std)
    
    # Trouver n_components pour variance_threshold
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    logger.info(f"PCA: {n_components} components pour {variance_threshold*100:.1f}% variance")
    
    # Refit avec n_components optimal
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)
    X_test_pca = pca.transform(X_test_std)
    
    return X_train_pca, X_val_pca, X_test_pca, pca
```

### Paramètres Recommandés pour Votre Dataset

```yaml
# Dans params.yaml, ajoutez:
pca:
  enabled: true
  variance_threshold: 0.95  # Conserver 95% de la variance
  max_components: 500  # Maximum 500 components
```

### Avantages/Inconvénients de PCA

**Avantages**:
- Réduit drastiquement la dimensionnalité (2352 → ~100-300)
- Élimine la multicolinéarité
- Accélère l'entraînement
- Réduit l'overfitting

**Inconvénients**:
- Perte d'interprétabilité (components = combinaisons linéaires)
- Ne capture que les relations linéaires
- Peut perdre des informations importantes pour la classification

---

## Gestion du Déséquilibre des Classes

### Situation Actuelle

- Classe 0 (nv): 1099 samples (12.4%)
- Classe 1 (mel): 6705 samples (75.6%) - MAJORITAIRE
- Classe 2 (bcc): 1113 samples (12.6%)

**Ratio déséquilibre**: ~6:1 entre majoritaire et minoritaires

### Stratégies de Gestion

#### 1. **Class Weights (déjà implémenté)**

```python
# Votre implémentation actuelle dans src/class_weights.py
def compute_balanced_class_weights(
    y: np.ndarray,
    *,
    rare_classes: list[int] | None = None,
    rare_class_boost: float = 1.2,  # Déjà réduit de 1.5 à 1.2
) -> dict[int, float]:
    """
    Poids balanced sklearn avec boost pour classes rares.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    w_map = {int(c): float(w) for c, w in zip(classes, weights)}
    
    if rare_classes and rare_class_boost != 1.0:
        for cls in rare_classes:
            cls = int(cls)
            if cls in w_map:
                w_map[cls] *= float(rare_class_boost)
    
    return w_map
```

**Amélioration possible**: Ajuster le boost dynamiquement

```python
def compute_adaptive_class_weights(
    y: np.ndarray,
    target_ratio: float = 2.0
) -> dict[int, float]:
    """
    Calcule des poids adaptatifs pour atteindre un ratio cible.
    
    Args:
        y: Labels
        target_ratio: Ratio cible entre majoritaire et minoritaires
    
    Returns:
        Dict {class: weight}
    """
    classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(classes, counts))
    
    max_count = max(counts)
    min_count = min(counts)
    
    # Calculer les poids pour atteindre target_ratio
    weights = {}
    for cls, count in class_counts.items():
        # Poids inversement proportionnel à la fréquence
        weights[int(cls)] = max_count / (count * target_ratio / (max_count / min_count))
    
    # Normaliser
    mean_weight = np.mean(list(weights.values()))
    weights = {cls: w / mean_weight for cls, w in weights.items()}
    
    return weights
```

#### 2. **SMOTE (Synthetic Minority Oversampling)**

```python
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: str | dict = "auto",
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique SMOTE pour générer des samples synthétiques des classes minoritaires.
    
    Args:
        X_train, y_train: Données d'entraînement
        sampling_strategy: 'auto' ou dict {class: ratio}
        random_state: Seed pour reproductibilité
    
    Returns:
        X_train_smote, y_train_smote
    """
    # SMOTE standard
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5  # Nombre de voisins pour générer samples
    )
    
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    logger.info(f"SMOTE: {len(X_train)} -> {len(X_train_smote)} samples")
    logger.info(f"Distribution après SMOTE: {np.bincount(y_train_smote)}")
    
    return X_train_smote, y_train_smote


def apply_smote_tomek(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Combinaison SMOTE + Tomek links pour un meilleur équilibrage.
    SMOTE: Oversampling des minoritaires
    Tomek: Undersampling des majoritaires (supprime les samples ambigus)
    """
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
    
    logger.info(f"SMOTE+Tomek: {len(X_train)} -> {len(X_train_balanced)} samples")
    
    return X_train_balanced, y_train_balanced
```

#### 3. **Undersampling de la Classe Majoritaire**

```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

def apply_undersampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: dict = None,
    random_state: int = 42
) -> tuple:
    """
    Undersampling de la classe majoritaire.
    
    Args:
        sampling_strategy: Dict {class: n_samples_target}
                          Ex: {0: 1099, 1: 2000, 2: 1113} réduit classe 1 à 2000
    """
    if sampling_strategy is None:
        # Stratégie par défaut: réduire majoritaire à 2x la minoritaire
        classes, counts = np.unique(y_train, return_counts=True)
        min_count = min(counts)
        sampling_strategy = {int(cls): min_count * 2 for cls in classes}
    
    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)
    
    logger.info(f"Undersampling: {len(X_train)} -> {len(X_train_under)} samples")
    
    return X_train_under, y_train_under
```

#### 4. **Combinaison Over/Under Sampling**

```python
from imblearn.combine import SMOTEENN

def apply_smoteenn(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Combinaison SMOTE (oversampling) + ENN (undersampling).
    ENN (Edited Nearest Neighbours) supprime les samples mal classés par leurs voisins.
    """
    smoteenn = SMOTEENN(random_state=42)
    X_train_balanced, y_train_balanced = smoteenn.fit_resample(X_train, y_train)
    
    logger.info(f"SMOTE+ENN: {len(X_train)} -> {len(X_train_balanced)} samples")
    
    return X_train_balanced, y_train_balanced
```

### Recommandation pour Votre Projet

```python
# Dans src/preprocessing.py, modifiez apply_imbalance_method:

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

def apply_imbalance_method(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "smote",  # "none", "smote", "smote_tomek", "undersample"
    sampling_strategy: str | dict = "auto"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique une méthode de gestion du déséquilibre.
    """
    if method == "none":
        logger.info("Using original training data without resampling")
        return X_train, y_train
    
    elif method == "smote":
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE applied: {len(X_train)} -> {len(X_train_bal)} samples")
        return X_train_bal, y_train_bal
    
    elif method == "smote_tomek":
        smote_tomek = SMOTETomek(random_state=42)
        X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)
        logger.info(f"SMOTE+Tomek applied: {len(X_train)} -> {len(X_train_bal)} samples")
        return X_train_bal, y_train_bal
    
    elif method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_train_bal, y_train_bal = undersampler.fit_resample(X_train, y_train)
        logger.info(f"Undersampling applied: {len(X_train)} -> {len(X_train_bal)} samples")
        return X_train_bal, y_train_bal
    
    else:
        raise ValueError(f"Unknown imbalance method: {method}")
```

### Configuration dans params.yaml

```yaml
preprocess:
  normalize_max: 255.0
  imbalance_method: smote_tomek  # none, smote, smote_tomek, undersample
  smote_strategy: auto  # auto ou dict {0: 1500, 1: 3000, 2: 1500}
```

---

## Optimisation des Hyperparamètres

### Votre Implémentation Actuelle (Optuna)

Vous avez déjà une implémentation avec Optuna dans `src/hyperparameter_tuning.py`. Voici des améliorations:

#### 1. **Espaces de Recherche Plus Larges**

```python
def objective_logistic(trial):
    """Objective amélioré pour LogisticRegression."""
    C = trial.suggest_float("C", 0.001, 100.0, log=True)
    max_iter = trial.suggest_categorical("max_iter", [1000, 3000, 5000, 10000])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    solver = "saga"  # Compatible avec toutes les pénalités
    
    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    params = load_params()
    processed_dir = ROOT / params["data"]["processed_dir"]
    bundle = load_processed(processed_dir)
    
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    
    from src.class_weights import compute_balanced_class_weights
    imb = params.get("imbalance", {})
    class_weights = compute_balanced_class_weights(
        y_train,
        rare_classes=imb.get("rare_classes"),
        rare_class_boost=float(imb.get("rare_class_boost", 1.0)),
    )
    
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        l1_ratio=l1_ratio,
        class_weight=class_weights,
        solver=solver,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=42,
    )
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        return f1
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 0.0


def objective_rf(trial):
    """Objective amélioré pour RandomForest."""
    n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300, 500, 700])
    max_depth = trial.suggest_categorical("max_depth", [5, 10, 15, 20, 25, None])
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 5, 10, 20])
    min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10, 20])
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    
    params = load_params()
    processed_dir = ROOT / params["data"]["processed_dir"]
    bundle = load_processed(processed_dir)
    
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        return f1
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 0.0


def objective_xgb(trial):
    """Objective amélioré pour XGBoost."""
    n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300, 500, 700, 1000])
    max_depth = trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8, 10])
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 2.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.1, 5.0)
    min_child_weight = trial.suggest_categorical("min_child_weight", [1, 3, 5, 7])
    gamma = trial.suggest_float("gamma", 0.0, 1.0)
    
    params = load_params()
    processed_dir = ROOT / params["data"]["processed_dir"]
    bundle = load_processed(processed_dir)
    
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    
    from src.class_weights import compute_balanced_class_weights, sample_weights_from_class_dict
    imb = params.get("imbalance", {})
    class_weights = compute_balanced_class_weights(
        y_train,
        rare_classes=imb.get("rare_classes"),
        rare_class_boost=float(imb.get("rare_class_boost", 1.0)),
    )
    sw = sample_weights_from_class_dict(y_train, class_weights)
    
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss",
    )
    
    try:
        model.fit(X_train, y_train, sample_weight=sw)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        return f1
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 0.0
```

#### 2. **Cross-Validation pour Évaluation Robuste**

```python
def objective_with_cv(trial, model_name: str, n_folds: int = 5):
    """
    Objective avec cross-validation pour une évaluation plus robuste.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    
    params = load_params()
    processed_dir = ROOT / params["data"]["processed_dir"]
    bundle = load_processed(processed_dir)
    
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    
    # Suggérer les hyperparamètres selon le modèle
    if model_name == "logistic_regression":
        C = trial.suggest_float("C", 0.001, 100.0, log=True)
        max_iter = 3000
        model = LogisticRegression(C=C, max_iter=max_iter, solver="saga", random_state=42)
    elif model_name == "random_forest":
        n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300, 500])
        max_depth = trial.suggest_categorical("max_depth", [5, 10, 15, 20, None])
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_name == "xgboost":
        n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300, 500])
        max_depth = trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)
        except Exception as e:
            logger.warning(f"CV fold failed: {e}")
            return 0.0
    
    return np.mean(f1_scores)
```

#### 3. **Early Stopping pour XGBoost**

```python
def objective_xgb_early_stopping(trial):
    """
    XGBoost avec early stopping pour éviter l'overfitting.
    """
    params = load_params()
    processed_dir = ROOT / params["data"]["processed_dir"]
    bundle = load_processed(processed_dir)
    
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    
    n_estimators = trial.suggest_categorical("n_estimators", [500, 1000, 2000])
    max_depth = trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 2.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.1, 5.0)
    
    from src.class_weights import compute_balanced_class_weights, sample_weights_from_class_dict
    imb = params.get("imbalance", {})
    class_weights = compute_balanced_class_weights(
        y_train,
        rare_classes=imb.get("rare_classes"),
        rare_class_boost=float(imb.get("rare_class_boost", 1.0)),
    )
    sw = sample_weights_from_class_dict(y_train, class_weights)
    
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss",
    )
    
    try:
        # Early stopping sur validation set
        model.fit(
            X_train, y_train,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Prédiction avec le meilleur nombre d'arbres
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        return f1
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 0.0
```

### Commande pour Lancer le Tuning

```bash
# Tuning XGBoost avec 100 trials
python -m src.hyperparameter_tuning --model xgboost --trials 100

# Tuning Random Forest avec 50 trials
python -m src.hyperparameter_tuning --model random_forest --trials 50

# Tuning Logistic Regression avec 50 trials
python -m src.hyperparameter_tuning --model logistic_regression --trials 50
```

---

## Sélection de Features

### Pourquoi la Sélection de Features?

1. **Réduit l'overfitting**: Moins de features = moins de bruit
2. **Accélère l'entraînement**: Moins de données à traiter
3. **Améliore l'interprétabilité**: Features plus pertinentes
4. **Élimine la redondance**: Supprime les features corrélées

### Méthodes de Sélection

#### 1. **Variance Threshold**

```python
from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    threshold: float = 0.01
) -> tuple:
    """
    Supprime les features avec variance inférieure au seuil.
    Utile pour éliminer les pixels constants ou quasi-constants.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    n_features_removed = X_train.shape[1] - X_train_sel.shape[1]
    logger.info(f"Variance Threshold: {n_features_removed} features removed")
    
    return X_train_sel, X_val_sel, X_test_sel, selector
```

#### 2. **SelectKBest avec ANOVA F-value**

```python
from sklearn.feature_selection import SelectKBest, f_classif

def select_k_best(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    k: int = 500
) -> tuple:
    """
    Sélectionne les k meilleures features selon le test ANOVA F-value.
    """
    selector = SelectKBest(f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    logger.info(f"SelectKBest: {k} features selected from {X_train.shape[1]}")
    
    return X_train_sel, X_val_sel, X_test_sel, selector
```

#### 3. **Recursive Feature Elimination (RFE)**

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def rfe_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    n_features_to_select: int = 500,
    estimator: str = "rf"
) -> tuple:
    """
    Recursive Feature Elimination avec un estimateur.
    """
    if estimator == "rf":
        base_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif estimator == "logistic":
        from sklearn.linear_model import LogisticRegression
        base_estimator = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    
    selector = RFE(
        estimator=base_estimator,
        n_features_to_select=n_features_to_select,
        step=0.1  # Élimine 10% des features à chaque itération
    )
    
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    logger.info(f"RFE: {n_features_to_select} features selected from {X_train.shape[1]}")
    
    return X_train_sel, X_val_sel, X_test_sel, selector
```

#### 4. **Feature Importance avec Tree-based Models**

```python
def tree_based_feature_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    threshold: float = 0.001,
    model: str = "rf"
) -> tuple:
    """
    Sélectionne les features basées sur l'importance des arbres.
    """
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif model == "xgb":
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    estimator.fit(X_train, y_train)
    
    # Sélectionner les features avec importance > threshold
    importances = estimator.feature_importances_
    selected_mask = importances > threshold
    selected_indices = np.where(selected_mask)[0]
    
    X_train_sel = X_train[:, selected_mask]
    X_val_sel = X_val[:, selected_mask]
    X_test_sel = X_test[:, selected_mask]
    
    logger.info(f"Tree-based selection: {len(selected_indices)} features selected (threshold={threshold})")
    
    return X_train_sel, X_val_sel, X_test_sel, selected_mask
```

#### 5. **SelectFromModel avec L1 (Lasso)**

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

def l1_feature_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    C: float = 0.1
) -> tuple:
    """
    Sélection de features avec L1 regularization (Lasso).
    L1 tend à produire des coefficients nuls → sélection automatique.
    """
    # Logistic Regression avec pénalité L1
    estimator = LogisticRegression(
        penalty="l1",
        C=C,
        solver="saga",
        multi_class="multinomial",
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    selector = SelectFromModel(estimator, threshold="median")
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    logger.info(f"L1 selection: {X_train_sel.shape[1]} features selected from {X_train.shape[1]}")
    
    return X_train_sel, X_val_sel, X_test_sel, selector
```

### Pipeline Complet de Feature Selection

```python
def feature_selection_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    method: str = "variance_then_kbest",
    **kwargs
) -> tuple:
    """
    Pipeline complet de sélection de features.
    
    Methods:
    - variance_then_kbest: Variance threshold puis SelectKBest
    - rfe: Recursive Feature Elimination
    - tree_based: Basé sur l'importance des arbres
    - l1: L1 regularization
    """
    X_tr, X_va, X_te = X_train, X_val, X_test
    
    if method == "variance_then_kbest":
        # Étape 1: Variance threshold
        X_tr, X_va, X_te, var_selector = variance_threshold_selection(
            X_tr, X_va, X_te, threshold=kwargs.get("var_threshold", 0.01)
        )
        
        # Étape 2: SelectKBest
        X_tr, X_va, X_te, kbest_selector = select_k_best(
            X_tr, X_va, X_te, y_train, k=kwargs.get("k", 500)
        )
        
        return X_tr, X_va, X_te, {"variance": var_selector, "kbest": kbest_selector}
    
    elif method == "rfe":
        return rfe_selection(
            X_tr, X_va, X_te, y_train,
            n_features_to_select=kwargs.get("n_features", 500),
            estimator=kwargs.get("estimator", "rf")
        )
    
    elif method == "tree_based":
        return tree_based_feature_selection(
            X_tr, X_va, X_te, y_train,
            threshold=kwargs.get("threshold", 0.001),
            model=kwargs.get("model", "rf")
        )
    
    elif method == "l1":
        return l1_feature_selection(
            X_tr, X_va, X_te, y_train,
            C=kwargs.get("C", 0.1)
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Configuration dans params.yaml

```yaml
feature_selection:
  enabled: true
  method: variance_then_kbest  # variance_then_kbest, rfe, tree_based, l1
  var_threshold: 0.01
  k: 500
  n_features: 500
  threshold: 0.001
```

---

## Amélioration des Métriques (Precision, Recall, F1)

### Comprendre les Métriques

- **Precision**: De toutes les prédictions positives, combien sont vraiment positives?
- **Recall**: De toutes les vraies positives, combien ont été correctement prédites?
- **F1-score**: Moyenne harmonique de precision et recall

### Stratégies d'Amélioration

#### 1. **Optimisation des Seuils de Décision**

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_metric: str = "f1",
    class_id: int = None
) -> float:
    """
    Optimise le seuil de décision pour une métrique cible.
    
    Args:
        y_true: Labels vrais
        y_proba: Probabilités prédites (shape: n_samples, n_classes)
        target_metric: 'f1', 'precision', 'recall'
        class_id: Classe cible (None = macro average)
    
    Returns:
        Optimal threshold
    """
    if class_id is not None:
        # Binary case pour une classe spécifique
        y_binary = (y_true == class_id).astype(int)
        y_score = y_proba[:, class_id]
        
        precision, recall, thresholds = precision_recall_curve(y_binary, y_score)
        
        if target_metric == "f1":
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
        elif target_metric == "precision":
            best_idx = np.argmax(precision)
            best_threshold = thresholds[best_idx]
        elif target_metric == "recall":
            best_idx = np.argmax(recall)
            best_threshold = thresholds[best_idx]
        
        return best_threshold
    else:
        # Multi-class: optimiser seuil par classe
        thresholds = []
        for cls in np.unique(y_true):
            thresh = optimize_threshold(y_true, y_proba, target_metric, class_id=cls)
            thresholds.append(thresh)
        
        return np.mean(thresholds)


def predict_with_threshold(
    model,
    X: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Prédiction avec seuil personnalisé.
    """
    y_proba = model.predict_proba(X)
    
    # Appliquer le seuil
    y_pred = np.argmax(y_proba, axis=1)
    
    # Pour les cas où la probabilité max est sous le seuil, utiliser "rejet"
    max_probas = np.max(y_proba, axis=1)
    uncertain_mask = max_probas < threshold
    
    # Optionnel: marquer les cas incertains
    # y_pred[uncertain_mask] = -1  # -1 = "incertain"
    
    return y_pred
```

#### 2. **Cost-Sensitive Learning**

```python
def create_cost_matrix():
    """
    Crée une matrice de coût pour la classification médicale.
    Faux négatifs (manquer un mélanome) = coût très élevé
    """
    # Classes: 0=nv, 1=mel, 2=bcc
    # Rows: vraie classe, Cols: prédite
    cost_matrix = np.array([
        [0, 5, 3],   # nv prédit comme mel (5), bcc (3)
        [10, 0, 8],  # mel prédit comme nv (10 - TRÈS GRAVE), bcc (8)
        [4, 6, 0]    # bcc prédit comme nv (4), mel (6)
    ])
    return cost_matrix


def custom_loss(y_true, y_pred, cost_matrix):
    """
    Calcule le coût total des prédictions.
    """
    total_cost = 0
    for true, pred in zip(y_true, y_pred):
        total_cost += cost_matrix[true, pred]
    return total_cost
```

#### 3. **Focal Loss (pour XGBoost custom)**

```python
import xgboost as xgb

def focal_loss_objective(preds, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss objective pour XGBoost.
    Réduit la contribution des samples faciles, focus sur les difficiles.
    """
    labels = dtrain.get_label()
    n_classes = len(np.unique(labels))
    
    # Reshape preds pour multi-class
    preds = preds.reshape(-1, n_classes)
    
    # Softmax
    preds_softmax = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    
    # One-hot encoding des labels
    labels_onehot = np.zeros_like(preds_softmax)
    labels_onehot[np.arange(len(labels)), labels] = 1
    
    # Focal loss
    pt = np.sum(labels_onehot * preds_softmax, axis=1)
    focal_loss = -alpha * (1 - pt) ** gamma * np.log(pt + 1e-8)
    
    grad = (preds_softmax - labels_onehot) * (alpha * gamma * (1 - pt) ** (gamma - 1) * np.log(pt + 1e-8) - (1 - pt) ** gamma)
    hess = preds_softmax * (1 - preds_softmax)
    
    return grad.flatten(), hess.flatten()
```

#### 4. **Class-Specific Metrics Analysis**

```python
def analyze_class_specific_performance(y_true, y_pred, y_proba, label_encoder):
    """
    Analyse détaillée des performances par classe.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=[str(c) for c in label_encoder.classes_],
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Analyse par classe
    class_analysis = {}
    for i, class_name in enumerate(label_encoder.classes_):
        class_analysis[str(class_name)] = {
            "precision": report[str(class_name)]["precision"],
            "recall": report[str(class_name)]["recall"],
            "f1": report[str(class_name)]["f1-score"],
            "support": report[str(class_name)]["support"],
            "most_confused_with": None,
            "confusion_rate": 0.0
        }
        
        # Trouver la classe la plus confondue
        row = cm[i]
        row[i] = 0  # Ignorer la diagonale
        if row.sum() > 0:
            most_confused_idx = np.argmax(row)
            most_confused_class = label_encoder.classes_[most_confused_idx]
            confusion_rate = row[most_confused_idx] / row.sum()
            
            class_analysis[str(class_name)]["most_confused_with"] = str(most_confused_class)
            class_analysis[str(class_name)]["confusion_rate"] = confusion_rate
    
    return class_analysis
```

---

## Réduction de l'Overfitting

### Signes d'Overfitting

- Accuracy train >> Accuracy test
- Loss train << Loss test
- Très bonne performance sur validation, mauvaise sur test

### Stratégies de Réduction

#### 1. **Augmentation de la Régularisation**

```python
# Logistic Regression
log_reg = LogisticRegression(
    C=0.1,  # Plus C est petit, plus la régularisation est forte
    penalty="l2",
    max_iter=1000
)

# Random Forest
rf = RandomForestClassifier(
    max_depth=10,  # Réduire la profondeur
    min_samples_leaf=4,  # Augmenter
    min_samples_split=10,  # Augmenter
    max_features="sqrt"  # Réduire les features par arbre
)

# XGBoost
xgb = XGBClassifier(
    reg_alpha=1.0,  # L1 regularization
    reg_lambda=3.0,  # L2 regularization
    max_depth=6,  # Réduire la profondeur
    learning_rate=0.01,  # Réduire le learning rate
    subsample=0.7,  # Subsampling des rows
    colsample_bytree=0.7  # Subsampling des features
)
```

#### 2. **Cross-Validation**

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

def cross_validation_evaluation(model, X, y, cv=5):
    """
    Évaluation avec cross-validation pour estimer la vraie performance.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        model, X, y,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1
    )
    
    print(f"CV F1 Macro: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return scores
```

#### 3. **Early Stopping (XGBoost)**

```python
def train_xgb_with_early_stopping(X_train, y_train, X_val, y_val):
    """
    XGBoost avec early stopping.
    """
    model = XGBClassifier(
        n_estimators=1000,  # Maximum d'arbres
        learning_rate=0.05,
        max_depth=6,
        eval_metric="mlogloss",
        early_stopping_rounds=50,  # Arrêter si pas d'amélioration après 50 rounds
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score}")
    
    return model
```

#### 4. **Dropout (pour réseaux de neurones, pas applicable ici)**

Non applicable pour les modèles classiques, mais mentionné pour completeness.

#### 5. **Bagging et Bootstrap**

```python
from sklearn.ensemble import BaggingClassifier

def bagging_classifier(base_model, X_train, y_train, n_estimators=10):
    """
    Bagging pour réduire la variance.
    """
    bagging = BaggingClassifier(
        base_estimator=base_model,
        n_estimators=n_estimators,
        max_samples=0.8,  # 80% des samples par bag
        max_features=0.8,  # 80% des features par bag
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    bagging.fit(X_train, y_train)
    
    return bagging
```

#### 6. **Réduction de la Complexité du Modèle**

```python
# Simplifier les modèles
log_reg_simple = LogisticRegression(C=1.0, max_iter=500)

rf_simple = RandomForestClassifier(
    n_estimators=100,  # Réduit
    max_depth=8,  # Plus shallow
    min_samples_leaf=5  # Plus de contraintes
)

xgb_simple = XGBClassifier(
    n_estimators=200,  # Réduit
    max_depth=4,  # Plus shallow
    learning_rate=0.1  # Plus simple
)
```

---

## Amélioration de la Matrice de Confusion

### Analyse de la Matrice de Confusion

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def detailed_confusion_analysis(y_true, y_pred, label_encoder):
    """
    Analyse détaillée de la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = label_encoder.classes_
    
    # Normalisation par row (recall par classe)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print("Matrice de confusion (brute):")
    print(cm)
    print("\nMatrice de confusion (normalisée):")
    print(cm_normalized)
    
    # Analyse des confusions principales
    print("\nPrincipales confusions:")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / cm[i].sum()
                print(f"{classes[i]} -> {classes[j]}: {cm[i, j]} cas ({confusion_rate*100:.1f}%)")
    
    return cm, cm_normalized


def plot_confusion_with_annotations(y_true, y_pred, label_encoder, save_path=None):
    """
    Matrice de confusion avec annotations détaillées.
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = label_encoder.classes_
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot matrice
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    # Ajouter les pourcentages
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                percentage = cm[i, j] / cm[i].sum() * 100
                ax.text(j, i, f'\n({percentage:.1f}%)',
                       ha='center', va='center', color='red', fontsize=8)
    
    plt.title('Matrice de Confusion avec Pourcentages')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Stratégies pour Réduire la Confusion

#### 1. **Feature Engineering Spécifique aux Classes Confondues**

```python
def create_discriminative_features(X):
    """
    Crée des features spécifiques pour discriminer les classes confondues.
    Pour les images: statistiques de couleur, texture, etc.
    """
    n_samples = X.shape[0]
    
    # Supposons X est (n_samples, height*width*channels)
    # Reshape en (n_samples, height, width, channels)
    height = width = 28
    channels = 3
    X_reshaped = X.reshape(n_samples, height, width, channels)
    
    new_features = []
    
    for i in range(n_samples):
        img = X_reshaped[i]
        
        # Features de couleur
        mean_r = img[:, :, 0].mean()
        mean_g = img[:, :, 1].mean()
        mean_b = img[:, :, 2].mean()
        std_r = img[:, :, 0].std()
        std_g = img[:, :, 1].std()
        std_b = img[:, :, 2].std()
        
        # Features de texture (simple)
        # Gradient horizontal et vertical
        grad_h = np.abs(np.diff(img, axis=1)).mean()
        grad_v = np.abs(np.diff(img, axis=0)).mean()
        
        # Contraste
        contrast = img.std()
        
        features = [
            mean_r, mean_g, mean_b,
            std_r, std_g, std_b,
            grad_h, grad_v,
            contrast
        ]
        
        new_features.append(features)
    
    new_features = np.array(new_features)
    
    # Concaténer avec les features originales
    X_enhanced = np.hstack([X, new_features])
    
    return X_enhanced
```

#### 2. **One-vs-Rest pour Classes Problématiques**

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

def one_vs_rest_for_confused_classes(X_train, y_train, confused_pairs):
    """
    Entraîne des classifieurs One-vs-Rest pour les paires de classes confondues.
    
    confused_pairs: list de tuples [(class_a, class_b), ...]
    """
    ovr_models = {}
    
    for class_a, class_b in confused_pairs:
        # Filtrer les données pour ces deux classes seulement
        mask = (y_train == class_a) | (y_train == class_b)
        X_pair = X_train[mask]
        y_pair = y_train[mask]
        
        # Remapper à 0 et 1
        y_pair_binary = (y_pair == class_b).astype(int)
        
        # Entraîner un classifieur binaire
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_pair, y_pair_binary)
        
        ovr_models[(class_a, class_b)] = model
        
        print(f"Trained OvR for {class_a} vs {class_b}")
    
    return ovr_models
```

#### 3. **Ensemble avec Vote Pondéré par Classe**

```python
def class_weighted_voting(models, X, class_weights):
    """
    Vote pondéré où chaque modèle a un poids différent selon la classe.
    """
    probas = []
    
    for model in models:
        if hasattr(model, 'predict_proba'):
            probas.append(model.predict_proba(X))
    
    # Moyenne pondérée par classe
    weighted_proba = np.zeros_like(probas[0])
    
    for i, proba in enumerate(probas):
        for cls in range(proba.shape[1]):
            weighted_proba[:, cls] += proba[:, cls] * class_weights.get(cls, 1.0)
    
    weighted_proba /= len(probas)
    
    return np.argmax(weighted_proba, axis=1)
```

#### 4. **Post-processing avec Règles Métier**

```python
def apply_medical_rules(y_pred, y_proba, label_encoder):
    """
    Applique des règles médicales pour corriger les prédictions.
    """
    y_pred_corrected = y_pred.copy()
    
    # Règle 1: Si probabilité de mel est élevée mais nv est prédit,
    # et la confiance est faible, reconsidérer
    mel_idx = list(label_encoder.classes_).index('mel') if 'mel' in label_encoder.classes_ else 1
    nv_idx = list(label_encoder.classes_).index('nv') if 'nv' in label_encoder.classes_ else 0
    
    for i in range(len(y_pred)):
        if y_pred[i] == nv_idx and y_proba[i, mel_idx] > 0.3:
            # Si probabilité de mel > 30% et nv prédit
            if y_proba[i].max() < 0.6:  # Confiance faible
                # Reconsidérer: prédire mel si sa probabilité est la 2ème plus haute
                sorted_indices = np.argsort(y_proba[i])[::-1]
                if sorted_indices[1] == mel_idx:
                    y_pred_corrected[i] = mel_idx
    
    return y_pred_corrected
```

---

## Exemples de Code Python

### Pipeline Complet avec Toutes les Améliorations

```python
"""
Pipeline complet pour la classification du cancer de la peau avec:
- Normalisation avancée
- PCA
- Gestion du déséquilibre (SMOTE)
- Sélection de features
- Optimisation des hyperparamètres
- Évaluation détaillée
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinCancerPipeline:
    """Pipeline complet pour la classification du cancer de la peau."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        self.label_encoder = None
        self.models = {}
        
    def load_data(self, csv_path):
        """Charge les données depuis CSV."""
        logger.info(f"Chargement des données depuis {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Séparer features et labels
        label_col = 'label' if 'label' in df.columns else df.columns[-1]
        X = df.drop(columns=[label_col]).values
        y = df[label_col].values
        
        logger.info(f"Données chargées: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def encode_labels(self, y):
        """Encode et remappe les labels."""
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        
        # Remap à 0, 1, 2 si 3 classes
        unique_classes = sorted(np.unique(y_enc))
        if len(unique_classes) == 3:
            mapping = {old: new for new, old in enumerate(unique_classes)}
            y_enc = np.array([mapping[label] for label in y_enc], dtype=np.int32)
            self.label_encoder.classes_ = np.array([0, 1, 2], dtype=int)
            logger.info(f"Labels remapped: {unique_classes} -> [0, 1, 2]")
        
        return y_enc
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
        """Split stratifié train/val/test."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        val_relative = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=random_state
        )
        
        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize(self, X_train, X_val, X_test, method='standard'):
        """Normalisation des données."""
        logger.info(f"Normalisation avec méthode: {method}")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test)
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def apply_pca(self, X_train, X_val, X_test, variance_threshold=0.95, max_components=500):
        """Applique PCA."""
        logger.info(f"PCA avec variance_threshold={variance_threshold}")
        
        self.pca = PCA(n_components=min(max_components, X_train.shape[1]))
        self.pca.fit(X_train)
        
        # Trouver n_components pour variance_threshold
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        logger.info(f"PCA: {n_components} components pour {variance_threshold*100:.1f}% variance")
        
        # Refit avec n_components optimal
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train)
        X_val_pca = self.pca.transform(X_val)
        X_test_pca = self.pca.transform(X_test)
        
        return X_train_pca, X_val_pca, X_test_pca
    
    def handle_imbalance(self, X_train, y_train, method='smote_tomek'):
        """Gère le déséquilibre des classes."""
        logger.info(f"Gestion du déséquilibre avec méthode: {method}")
        
        if method == 'none':
            return X_train, y_train
        
        elif method == 'smote_tomek':
            smote_tomek = SMOTETomek(random_state=42)
            X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)
            logger.info(f"SMOTE+Tomek: {len(X_train)} -> {len(X_train_bal)} samples")
            return X_train_bal, y_train_bal
        
        else:
            raise ValueError(f"Unknown imbalance method: {method}")
    
    def select_features(self, X_train, X_val, X_test, y_train, method='variance_then_kbest', k=500):
        """Sélection de features."""
        logger.info(f"Sélection de features avec méthode: {method}")
        
        X_tr, X_va, X_te = X_train, X_val, X_test
        
        if method == 'variance_then_kbest':
            # Variance threshold
            var_selector = VarianceThreshold(threshold=0.01)
            X_tr = var_selector.fit_transform(X_tr)
            X_va = var_selector.transform(X_va)
            X_te = var_selector.transform(X_te)
            
            # SelectKBest
            kbest_selector = SelectKBest(f_classif, k=min(k, X_tr.shape[1]))
            X_tr = kbest_selector.fit_transform(X_tr, y_train)
            X_va = kbest_selector.transform(X_va)
            X_te = kbest_selector.transform(X_te)
            
            self.feature_selector = {'variance': var_selector, 'kbest': kbest_selector}
        
        logger.info(f"Features après sélection: {X_tr.shape[1]}")
        
        return X_tr, X_va, X_te
    
    def compute_class_weights(self, y_train, rare_classes=None, rare_class_boost=1.2):
        """Calcule les poids de classes."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        w_map = {int(c): float(w) for c, w in zip(classes, weights)}
        
        if rare_classes and rare_class_boost != 1.0:
            for cls in rare_classes:
                if cls in w_map:
                    w_map[cls] *= rare_class_boost
        
        logger.info(f"Class weights: {w_map}")
        return w_map
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val, class_weights):
        """Entraîne un modèle."""
        logger.info(f"Entraînement de {model_name}")
        
        if model_name == 'logistic_regression':
            model = LogisticRegression(
                C=0.5,
                max_iter=1000,
                class_weight=class_weights,
                solver='saga',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=4,
                min_samples_split=5,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        elif model_name == 'xgboost':
            from src.class_weights import sample_weights_from_class_dict
            sw = sample_weights_from_class_dict(y_train, class_weights)
            
            model = XGBClassifier(
                objective='multi:softprob',
                num_class=len(np.unique(y_train)),
                n_estimators=400,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train, sample_weight=sw)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.models[model_name] = model
        logger.info(f"{model_name} entraîné")
        
        return model
    
    def evaluate(self, model, X_test, y_test):
        """Évalue un modèle."""
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Métriques
        accuracy = (y_pred == y_test).mean()
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Rapport de classification
        report = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in self.label_encoder.classes_],
            output_dict=True,
            zero_division=0
        )
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
        
        return results
    
    def run_pipeline(self, csv_path):
        """Exécute le pipeline complet."""
        # 1. Charger les données
        X, y = self.load_data(csv_path)
        
        # 2. Encoder les labels
        y = self.encode_labels(y)
        
        # 3. Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 4. Normalisation
        X_train, X_val, X_test = self.normalize(X_train, X_val, X_test, method='standard')
        
        # 5. PCA
        X_train, X_val, X_test = self.apply_pca(X_train, X_val, X_test, variance_threshold=0.95)
        
        # 6. Gestion du déséquilibre
        X_train, y_train = self.handle_imbalance(X_train, y_train, method='smote_tomek')
        
        # 7. Sélection de features (optionnel, après PCA peut ne pas être nécessaire)
        # X_train, X_val, X_test = self.select_features(X_train, X_val, X_test, y_train)
        
        # 8. Calculer les poids de classes
        class_weights = self.compute_class_weights(y_train, rare_classes=[0, 2], rare_class_boost=1.2)
        
        # 9. Entraîner les modèles
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            model = self.train_model(model_name, X_train, y_train, X_val, y_val, class_weights)
            results = self.evaluate(model, X_test, y_test)
            print(f"\n{model_name} Results:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"F1 Macro: {results['f1_macro']:.4f}")
        
        return self.models


# Exemple d'utilisation
if __name__ == "__main__":
    config = {
        'csv_path': 'data/raw/hmnist_filtered.csv',
        'normalization': 'standard',
        'pca_variance': 0.95,
        'imbalance_method': 'smote_tomek',
        'rare_class_boost': 1.2
    }
    
    pipeline = SkinCancerPipeline(config)
    models = pipeline.run_pipeline(config['csv_path'])
```

---

## Comparaison des Modèles

### Logistic Regression

**Avantages**:
- **Interprétable**: Coefficients directement interprétables
- **Rapide**: Entraînement et prédiction rapides
- **Probabilités**: Produit des probabilités bien calibrées
- **Baseline**: Bon point de départ

**Inconvénients**:
- **Linéaire**: Ne capture que les relations linéaires
- **Sensible aux outliers**: Peut être affecté par des valeurs extrêmes
- **Multicolinéarité**: Performance dégradée si features corrélées
- **Limité pour images**: Ne capture pas les structures spatiales

**Pour votre projet**:
- **Performance**: Généralement la plus faible des 3
- **Interprétabilité**: Excellente pour comprendre quelles features sont importantes
- **Utilisation**: Bon pour baseline et analyse

### Random Forest

**Avantages**:
- **Non-linéaire**: Capture les relations complexes
- **Robuste**: Résistant aux outliers et overfitting (avec bons paramètres)
- **Feature importance**: Fournit l'importance des features
- **Pas de scaling**: Ne nécessite pas de normalisation
- **Parallélisable**: Entraînement parallèle possible

**Inconvénients**:
- **Lent**: Entraînement peut être lent avec beaucoup d'arbres
- **Mémoire**: Peut consommer beaucoup de mémoire
- **Overfitting**: Peut overfitter si pas régularisé
- **Probabilités**: Peut être mal calibré

**Pour votre projet**:
- **Performance**: Généralement meilleure que Logistic Regression
- **Robustesse**: Bon choix pour les données bruitées
- **Utilisation**: Bon compromis performance/complexité

### XGBoost

**Avantages**:
- **Performance**: Souvent le meilleur des 3 pour les données tabulaires
- **Régularisation**: L1 et L2 intégrées
- **Flexibilité**: Beaucoup d'hyperparamètres à tuner
- **Gestion du déséquilibre**: Supporte sample weights nativement
- **Vitesse**: Entraînement rapide avec histogram-based method

**Inconvénients**:
- **Complexité**: Beaucoup d'hyperparamètres à tuner
- **Overfitting**: Peut facilement overfitter si mal configuré
- **Sensibilité aux paramètres**: Performance varie beaucoup avec les paramètres
- **Moins interprétable**: Plus difficile à interpréter que Logistic Regression

**Pour votre projet**:
- **Performance**: Généralement le meilleur des 3
- **Flexibilité**: Très adaptable à votre problème
- **Utilisation**: Recommandé comme modèle principal

### Tableau Comparatif

| Critère | Logistic Regression | Random Forest | XGBoost |
|---------|---------------------|---------------|---------|
| Performance | Bas | Moyen | Haut |
| Vitesse d'entraînement | Très rapide | Lent | Rapide |
| Vitesse de prédiction | Très rapide | Moyen | Rapide |
| Interprétabilité | Excellente | Moyenne | Faible |
| Résistance au overfitting | Moyenne | Bonne | Moyenne (avec tuning) |
| Gestion du déséquilibre | Class weights | Class weights | Sample weights |
| Sensibilité au scaling | Oui | Non | Non |
| Capture de relations non-linéaires | Non | Oui | Oui |
| Feature importance | Coefficients | Gini/Entropy | Gain/Cover |
| Probabilités calibrées | Oui | Non | Non (par défaut) |
| Tuning requis | Faible | Moyen | Élevé |

### Recommandation pour Votre Projet

1. **XGBoost comme modèle principal**: Meilleure performance potentielle
2. **Random Forest comme backup**: Plus robuste, moins de tuning
3. **Logistic Regression pour analyse**: Interprétabilité et baseline
4. **Ensemble**: Combiner les 3 pour meilleures performances

---

## Limites des Modèles Classiques pour les Images Médicales

### 1. **Perte de l'Information Spatiale**

**Problème**: Les modèles classiques travaillent sur des features aplaties (1D), détruisant la structure 2D/3D des images.

**Conséquence**:
- Les relations entre pixels voisins sont perdues
- Les motifs locaux (textures, formes) ne sont pas capturés
- L'invariance à la translation n'existe pas

**Exemple**: Un mélanome peut être identifié par sa forme irrégulière, mais un modèle classique ne "voit" pas la forme, seulement les pixels individuels.

### 2. **Incapacité à Apprendre des Hiérarchies de Features**

**Problème**: Les CNN apprennent des hiérarchies (edges → textures → motifs → objets), les modèles classiques ne peuvent pas.

**Conséquence**:
- Impossible d'apprendre des features complexes automatiquement
- Nécessité de feature engineering manuel
- Limité à des features simples (statistiques de couleur, etc.)

### 3. **Sensibilité à la Localisation**

**Problème**: Les modèles classiques ne sont pas invariants à la translation/rotation.

**Conséquence**:
- Une lésion déplacée dans l'image sera traitée différemment
- Nécessité de data augmentation massive
- Sensible aux variations de position/angle

### 4. **Curse of Dimensionality**

**Problème**: 2352 features (28x28x3) pour seulement 8917 samples.

**Conséquence**:
- Ratio features/samples défavorable
- Overfitting très probable
- Beaucoup de bruit dans les features

**Règle empirique**: Il faut environ 10x plus de samples que de features pour un modèle classique robuste.

### 5. **Manque de Contexte Global**

**Problème**: Les modèles classiques considèrent chaque feature indépendamment.

**Conséquence**:
- Difficile de capturer des relations globales
- Les patterns complexes sont manqués
- Performance limitée sur des tâches de reconnaissance fine

### 6. **Limitations pour les Tâches Médicales Spécifiques**

**Tâches médicales nécessitant**:
- **Segmentation**: Délimiter précisément la lésion (impossible avec modèles classiques)
- **Détection de motifs subtils**: Variations de couleur mineures, asymétrie
- **Analyse de la texture**: Grain de la peau, irrégularités des bordures
- **Contexte anatomique**: Position de la lésion sur le corps

### 7. **Comparaison avec CNN**

| Aspect | Modèles Classiques | CNN |
|--------|-------------------|-----|
| Capture de structures spatiales | Non | Oui |
| Hiérarchie de features | Non | Oui |
| Invariance translation/rotation | Non | Oui (avec pooling/augmentation) |
| Feature engineering | Manuel | Automatique |
| Performance sur images | Limitée | Excellente |
| Interprétabilité | Élevée | Moyenne (avec techniques) |
| Données requises | Moins | Plus |
| Temps d'entraînement | Court | Long |

### Pourquoi Votre Projet Utilise des Modèles Classiques?

**Avantages de votre approche**:
1. **Rapidité**: Développement et entraînement plus rapides
2. **Simplicité**: Moins de complexité technique
3. **Interprétabilité**: Plus facile d'expliquer les décisions
4. **MLOps**: Plus facile à intégrer dans un pipeline MLOps
5. **Ressources**: Moins de ressources GPU requises

**Quand les modèles classiques sont suffisants**:
- Images de petite taille (28x28)
- Tâches de classification simple
- Données limitées
- Besoin d'interprétabilité
- Contraintes de ressources

### Améliorations Possibles Sans CNN

Même sans CNN, vous pouvez améliorer significativement:

1. **Feature Engineering Avancé**:
   - Statistiques de couleur (moyenne, std, skewness)
   - Features de texture (GLCM, LBP)
   - Features de forme (après segmentation)
   - Histogrammes de couleur

2. **Ensemble Learning**:
   - Combiner plusieurs modèles
   - Stacking avec métaclassifieur
   - Voting pondéré

3. **Data Augmentation**:
   - Rotation, flip, zoom
   - Variation de luminosité/contraste
   - Ajout de bruit

4. **Transfer Learning (si possible)**:
   - Utiliser des features pré-entraînées
   - Extraire des features avec un CNN pré-entraîné
   - Utiliser ces features avec vos modèles classiques

---

## Bonnes Pratiques MLOps

### 1. **Versioning des Données**

```yaml
# Utiliser DVC pour le versioning
# dvc.yaml
stages:
  prepare:
    cmd: python -m src.train --prepare-only
    deps:
      - data/raw/hmnist_filtered.csv
    outs:
      - data/processed/dataset.npz
```

### 2. **Expérimentation avec MLflow**

```python
# Votre projet utilise déjà MLflow - excellent!
# Bonnes pratiques additionnelles:

with mlflow.start_run():
    # Loguer tous les hyperparamètres
    mlflow.log_params({
        "model": model_name,
        "normalization": "standard",
        "pca_variance": 0.95,
        "imbalance_method": "smote_tomek"
    })
    
    # Loguer les métriques
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    })
    
    # Loguer les artefacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("classification_report.json")
    
    # Loguer le modèle
    mlflow.sklearn.log_model(model, "model")
```

### 3. **Configuration Centralisée**

```yaml
# params.yaml - déjà implémenté, excellent!
# Ajouter plus de paramètres:

feature_engineering:
  enabled: true
  color_stats: true
  texture_features: true
  
data_augmentation:
  enabled: false
  rotation_range: 15
  flip_horizontal: true
  
model_selection:
  strategy: "best_f1"  # best_f1, best_accuracy, ensemble
  cv_folds: 5
```

### 4. **Tests Automatisés**

```python
# tests/test_pipeline.py
import pytest
import numpy as np

def test_data_loading():
    """Test que les données se chargent correctement."""
    from src.ingestion import load_csv
    df = load_csv("data/raw/hmnist_filtered.csv")
    assert df is not None
    assert len(df) > 0

def test_normalization():
    """Test que la normalisation fonctionne."""
    from src.preprocessing import normalize_pixels
    X = np.random.randint(0, 256, (100, 2352))
    X_norm = normalize_pixels(X)
    assert X_norm.max() <= 1.0
    assert X_norm.min() >= 0.0

def test_class_weights():
    """Test le calcul des poids de classes."""
    from src.class_weights import compute_balanced_class_weights
    y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    weights = compute_balanced_class_weights(y)
    assert len(weights) == 3
    assert all(w > 0 for w in weights.values())
```

### 5. **Monitoring en Production**

```python
# src/monitoring/metrics.py
class ModelMonitor:
    """Monitoring des performances du modèle en production."""
    
    def __init__(self, model, reference_data):
        self.model = model
        self.reference_data = reference_data
        self.reference_predictions = model.predict(reference_data['X'])
    
    def check_drift(self, new_data, threshold=0.05):
        """Détecte le data drift."""
        from scipy import stats
        
        # Test de Kolmogorov-Smirnov pour détecter le drift
        drift_detected = False
        for i in range(new_data['X'].shape[1]):
            _, p_value = stats.ks_2samp(
                self.reference_data['X'][:, i],
                new_data['X'][:, i]
            )
            if p_value < threshold:
                drift_detected = True
                break
        
        return drift_detected
    
    def check_performance_degradation(self, new_X, new_y, threshold=0.1):
        """Détecte la dégradation des performances."""
        from sklearn.metrics import f1_score
        
        new_pred = self.model.predict(new_X)
        new_f1 = f1_score(new_y, new_pred, average='macro')
        
        ref_f1 = f1_score(
            self.reference_data['y'],
            self.reference_predictions,
            average='macro'
        )
        
        degradation = (ref_f1 - new_f1) / ref_f1
        
        return degradation > threshold
```

### 6. **CI/CD Pipeline**

```yaml
# .github/workflows/train.yml
name: Train Model

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: pytest tests/
    
    - name: Train model
      run: python -m src.train --force-prepare
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v2
      with:
        name: model-artifacts
        path: models/
```

### 7. **Documentation**

```markdown
# README.md

## Skin Cancer Classification MLOps Project

### Overview
Classification of skin lesions using classical ML models on CSV pixel data.

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python -m src.train --force-prepare
```

### Hyperparameter Tuning
```bash
python -m src.hyperparameter_tuning --model xgboost --trials 100
```

### MLflow UI
```bash
mlflow ui
```

### Project Structure
- `src/`: Source code
- `data/`: Data (raw and processed)
- `models/`: Trained models
- `reports/`: Reports and figures
- `tests/`: Unit tests

### Configuration
Edit `params.yaml` to modify hyperparameters and pipeline settings.
```

### 8. **Reproductibilité**

```python
# Toujours utiliser des seeds
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
# Dans src/config.py
RANDOM_SEED = 42
```

### 9. **Gestion des Secrets**

```bash
# .env
SKIN_CSV_PATH=/path/to/data/hmnist_filtered.csv
MLFLOW_TRACKING_URI=http://localhost:5000

# .gitignore
.env
```

### 10. **Logging Structuré**

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    """Formatter pour les logs en JSON."""
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName
        }
        return json.dumps(log_obj)

# Configuration
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

---

## Résumé et Plan d'Action

### Actions Immédiates (Priorité Haute)

1. **Activer la normalisation avancée**:
   - Remplacer la simple division par 255 par StandardScaler
   - Fit sur train uniquement, transformer val/test

2. **Appliquer PCA**:
   - Réduire de 2352 features à ~200-300 components
   - Conserver 95% de la variance

3. **Activer SMOTE+Tomek**:
   - Équilibrer les classes
   - Combinaison over/under sampling

4. **Réduire l'overfitting**:
   - Augmenter la régularisation (déjà fait dans params.yaml)
   - Utiliser early stopping pour XGBoost

### Actions Moyen Terme (Priorité Moyenne)

5. **Sélection de features**:
   - Variance threshold pour éliminer les pixels constants
   - SelectKBest pour garder les features les plus discriminantes

6. **Optimisation des hyperparamètres**:
   - Lancer Optuna avec plus de trials
   - Utiliser cross-validation pour évaluation robuste

7. **Ensemble methods**:
   - Combiner les 3 modèles avec voting pondéré
   - Essayer stacking

8. **Feature engineering**:
   - Ajouter des statistiques de couleur
   - Ajouter des features de texture

### Actions Long Terme (Priorité Basse)

9. **Monitoring en production**:
   - Implémenter le drift detection
   - Surveiller les performances en continu

10. **CI/CD**:
    - Automatiser le training
    - Tests automatiques

### Résultats Attendus

Avec ces améliorations, vous devriez observer:
- **F1 macro**: +5-15% d'amélioration
- **Matrice de confusion**: Réduction des confusions entre nv, mel, bcc
- **Stabilité**: Moins d'overfitting, meilleure généralisation
- **Interprétabilité**: Meilleure compréhension des features importantes

### Limites à Connaître

Même avec toutes ces améliorations, les modèles classiques ont des limites inhérentes pour la classification d'images:
- Performance plafonnée (~70-80% F1 macro au mieux)
- Incapacité à capturer des motifs visuels complexes
- Sensibilité aux variations d'acquisition

Si vous avez besoin de performances supérieures à 85-90%, envisagez:
- CNN (ResNet, EfficientNet)
- Transfer learning depuis ImageNet
- Architectures spécialisées pour images médicales

---

## Conclusion

Votre projet MLOps est bien structuré avec DVC, MLflow, et une séparation claire des responsabilités. Les problèmes de confusion entre nv, mel, et bcc sont typiques pour des modèles classiques appliqués à des données d'images.

Les solutions proposées couvrent:
- Normalisation avancée
- PCA pour réduction de dimension
- Gestion du déséquilibre avec SMOTE
- Sélection de features
- Optimisation des hyperparamètres
- Ensemble methods
- Amélioration des métriques

En appliquant ces améliorations de manière systématique et en suivant les bonnes pratiques MLOps, vous devriez pouvoir améliorer significativement les performances de vos modèles tout en maintenant une architecture reproductible et scalable.
