# Classification automatisée des lésions cutanées — Projet MLOps

## Présentation

Ce projet met en œuvre un **pipeline MLOps complet** pour la classification de lésions cutanées à partir de données de pixels (images 28×28 RGB aplaties en vecteurs de 2352 features). L'approche repose exclusivement sur des **modèles de machine learning classiques** (sans CNN ni deep learning) : Logistic Regression, Random Forest et XGBoost.

L'objectif est de couvrir l'ensemble du cycle de vie d'un modèle ML : préparation des données, entraînement, évaluation, suivi d'expériences, déploiement API, interface utilisateur et monitoring en production — avec reproductibilité, traçabilité et maintenabilité.

---

## Résumé des réalisations

### 1. Données et regroupement des classes

- Utilisation du dataset public **HAM10000** (`hmnist_28_28_RGB.csv`).
- Regroupement des **7 classes originales en 3 catégories cliniques** via `data/raw/dat.py` :
  - **Classe 0** — Nevus mélanocytaire (nv)
  - **Classe 1** — Lésions bénignes (bkl, df, vasc)
  - **Classe 2** — Lésions malignes / précancéreuses (mel, bcc, akiec)
- Dataset traité exporté : `data/raw/hmnist_3classes.csv`
- Visualisations et rapport de distribution : `data/visualisations/`
- Documentation de justification : [`docs/HAM10000_3CLASSES_JUSTIFICATION.md`](docs/HAM10000_3CLASSES_JUSTIFICATION.md)

### 2. Pipeline de prétraitement (`src/preprocessing.py`, `src/train.py`)

| Étape | Description |
|-------|-------------|
| Ingestion | Chargement CSV, validation, détection des valeurs manquantes |
| Normalisation simple | Division des pixels par 255 |
| Encodage des labels | Remapping séquentiel 0, 1, 2 |
| **Undersampling NV** | Limitation de la classe majoritaire (NV) à **3000 échantillons max** (`nv_max_samples` dans `params.yaml`), reproductible (`random_state=42`), **avant le split** pour éviter toute fuite de données |
| Split stratifié | Train 70 % / Validation 15 % / Test 15 % |
| Normalisation avancée | StandardScaler (ou MinMax / Robust) — **fit sur le train uniquement** |
| PCA (optionnel) | Réduction de dimension configurable (désactivée par défaut) |
| Rééquilibrage (optionnel) | SMOTE, SMOTE+Tomek ou undersampling **sur le train uniquement** |
| Sauvegarde | `data/processed/dataset.npz`, scaler, label encoder, métadonnées |

**Distribution après undersampling NV :**

| Classe | Avant cap | Après cap (max 3000 NV) |
|--------|-----------|-------------------------|
| 0 (nv) | ~6 705 | 3 000 |
| 1 (bénin) | ~1 356 | 1 356 |
| 2 (malin) | ~1 954 | 1 954 |
| **Total** | ~10 015 | **~6 310** |

### 3. Gestion du déséquilibre des classes

- **Undersampling ciblé NV** : réduit la domination de la classe majoritaire sans toucher aux classes minoritaires.
- **Class weights** (`src/class_weights.py`) : stratégie `balanced_minority` avec boost modéré des classes rares (1 et 2).
- **SMOTE / undersampling imblearn** : disponibles en complément, appliqués uniquement sur le jeu d'entraînement.
- Analyse documentée des confusions inter-classes (nv / mel / bcc) : [`docs/CONFUSION_ANALYSIS_AND_SOLUTIONS.md`](docs/CONFUSION_ANALYSIS_AND_SOLUTIONS.md)

### 4. Modèles et entraînement

Trois algorithmes comparés automatiquement :

| Modèle | Particularités |
|--------|----------------|
| **Logistic Regression** | Régularisation L2 (`C=0.5`), solver `saga`, class weights |
| **Random Forest** | 300 arbres, `max_depth=10`, feuilles min. 4 — limiter l'overfitting |
| **XGBoost** | 400 estimateurs, régularisation L1/L2 renforcée, `sample_weight` |

- Hyperparamètres centralisés dans [`params.yaml`](params.yaml)
- Sélection automatique du meilleur modèle (F1 macro sur validation)
- Artefacts sauvegardés dans `models/artifacts/`
- Module d'optimisation Optuna disponible : `src/hyperparameter_tuning.py`
- Ensembles avancés (voting pondéré, stacking) : `src/ensemble_methods.py`

### 5. Évaluation et diagnostic

- Métriques : accuracy, precision, recall, F1-score (macro / par classe), ROC-AUC
- Matrices de confusion exportées en PNG (`reports/figures/`)
- Rapports de classification JSON par modèle
- Diagnostic approfondi (`src/diagnostic.py`) : entropie des prédictions, analyse par classe, identification des confusions
- Comparaison des modèles : `reports/figures/model_comparison.csv`

### 6. Suivi MLOps avec MLflow

- Enregistrement des hyperparamètres, métriques et artefacts
- Comparaison des runs (`skin_lesion_3class`)
- Support du Model Registry (optionnel)
- Interface MLflow disponible via Docker (port 5000)

### 7. Déploiement et interface

**API REST FastAPI** (`src/api/main.py`) :

```bash
GET  /health    # État du service
POST /predict   # Prédiction à partir d'un vecteur de pixels
```

**Interface Streamlit** (`streamlit_app.py`) :
- Upload d'image JPG/PNG
- Redimensionnement 28×28, conversion RGB, extraction des pixels
- Envoi à l'API et affichage du résultat avec probabilités

### 8. Monitoring en production

Module `src/monitoring/` :
- Latence des prédictions
- Probabilité maximale et classe prédite
- Mean Absolute Drift par rapport à une référence
- Logs JSONL dans `reports/logs/`

### 9. Reproductibilité et industrialisation

| Outil | Rôle |
|-------|------|
| **DVC** | Pipeline `prepare` → `train` versionné (`dvc.yaml`) |
| **Docker** | Containerisation API + MLflow UI (`Dockerfile`, `docker-compose.yml`) |
| **GitHub Actions** | CI : lint Ruff, entraînement rapide, tests Pytest |
| **Pytest** | Tests unitaires (ingestion, preprocessing, API, class weights) |
| **params.yaml** | Configuration unique pour tout le pipeline |

---

## Architecture du projet

```
pfa/
├── data/
│   ├── raw/
│   │   ├── dat.py                  # Regroupement 7 → 3 classes
│   │   └── hmnist_3classes.csv     # Dataset prétraité
│   ├── processed/                  # Données splitées et normalisées
│   └── visualisations/             # Graphiques de distribution
├── docs/
│   ├── HAM10000_3CLASSES_JUSTIFICATION.md
│   └── CONFUSION_ANALYSIS_AND_SOLUTIONS.md
├── models/artifacts/               # Modèles entraînés (.joblib)
├── reports/
│   ├── figures/                    # Matrices de confusion, rapports
│   └── logs/                       # Monitoring JSONL
├── src/
│   ├── train.py                    # Orchestration prepare + train
│   ├── preprocessing.py            # Normalisation, split, undersampling NV
│   ├── ingestion.py                # Chargement et validation CSV
│   ├── class_weights.py            # Pondération des classes
│   ├── evaluation.py               # Métriques et visualisations
│   ├── diagnostic.py               # Analyse des confusions
│   ├── ensemble_methods.py         # Voting / Stacking
│   ├── hyperparameter_tuning.py    # Optimisation Optuna
│   ├── mlflow_utils.py             # Intégration MLflow
│   ├── inference.py                # Inférence locale
│   ├── api/main.py                 # API FastAPI
│   └── monitoring/                 # Métriques production
├── tests/                          # Tests unitaires
├── streamlit_app.py                # Interface utilisateur
├── params.yaml                     # Hyperparamètres centralisés
├── dvc.yaml                        # Pipeline DVC
├── Dockerfile
└── docker-compose.yml
```

---

## Démarrage rapide

### Prérequis

```bash
pip install -r requirements.txt
```

### 1. Générer le dataset 3 classes (si nécessaire)

```bash
python data/raw/dat.py
```

### 2. Préparer les données et entraîner

```bash
# Pipeline complet (prepare + train)
python -m src.train

# Forcer la régénération du preprocessing (ex. après changement de nv_max_samples)
python -m src.train --force-prepare

# Étapes séparées
python -m src.train --prepare-only --force-prepare
python -m src.train --train-only
```

### 3. Lancer l'API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Lancer l'interface Streamlit

```bash
streamlit run streamlit_app.py
```

### 5. Pipeline DVC

```bash
dvc repro
```

### 6. Docker

```bash
docker compose up --build
```

---

## Configuration clé (`params.yaml`)

```yaml
preprocess:
  nv_max_samples: 3000          # Cap NV (null pour désactiver)
  normalization_method: standard
  imbalance_method: none        # none | smote | smote_tomek | undersample

imbalance:
  strategy: balanced_minority
  rare_classes: [1, 2]
  rare_class_boost: 1.2

pca:
  enabled: false                # true pour activer la PCA
```

---

## Choix techniques importants

### Pourquoi des modèles classiques (sans CNN) ?

Les pixels bruts en 28×28 perdent la structure spatiale 2D. Les modèles tabulaires (LR, RF, XGBoost) restent pertinents pour un prototype MLOps rapide, interprétable et léger, mais ont des **limites intrinsèques** pour distinguer des lésions visuellement proches (nv vs mel vs bcc). Le projet compense partiellement via le regroupement en 3 classes, l'undersampling NV, les class weights et la régularisation.

### Prévention du data leakage

| Opération | Moment | Fit sur |
|-----------|--------|---------|
| Undersampling NV | Avant split | Dataset complet (sélection aléatoire) |
| StandardScaler | Après split | Train uniquement |
| SMOTE | Après split | Train uniquement |
| PCA | Après split | Train uniquement |
| Class weights | Entraînement | Train uniquement |

### Undersampling NV : placement dans le pipeline

L'undersampling est appliqué **après le remapping des labels** (NV = classe 0) et **avant le split train/val/test**, afin que les trois jeux reflètent la même distribution et que l'évaluation reste cohérente.

---

## Technologies

Python · Scikit-learn · XGBoost · Pandas · NumPy · imbalanced-learn · FastAPI · Streamlit · MLflow · Optuna · DVC · Docker · GitHub Actions · Pytest · Ruff

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## Avertissement médical

Ce projet est un **travail académique / de démonstration MLOps**. Il ne constitue en aucun cas un outil de diagnostic médical et ne doit pas être utilisé en conditions cliniques réelles.
