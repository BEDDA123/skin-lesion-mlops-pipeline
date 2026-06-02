# Classification automatisée des lésions cutanées via une approche MLOps

## Présentation du projet

Ce projet met en œuvre une solution complète de classification automatisée des lésions cutanées basée sur une approche MLOps.

L’objectif principal est de concevoir un pipeline de Machine Learning reproductible couvrant l’ensemble du cycle de vie d’un modèle, allant de la préparation des données jusqu’au déploiement en production.

Le projet intègre plusieurs étapes essentielles, notamment l’entraînement et l’évaluation de plusieurs modèles, le suivi des expérimentations via MLflow, le déploiement d’une API REST, la création d’une interface utilisateur, ainsi que la mise en place d’un système de monitoring en environnement de production.

Ainsi, l’ensemble du workflow permet d’assurer la reproductibilité, la traçabilité et la maintenabilité du système.

---

## Objectif

Développer un système intelligent capable de classifier automatiquement des lésions cutanées à partir de données de pixels extraites d’images dermatologiques et de fournir une prédiction accompagnée d’un niveau de confiance.

---

## Dataset

Le projet utilise le dataset HAM10000, une base de données publique largement utilisée dans le domaine de la dermatologie.

### Données utilisées

Les images sont représentées sous forme de vecteurs de pixels RGB :

* Taille des images : 28 × 28 pixels
* Nombre de canaux : 3 (RGB)
* Nombre total de caractéristiques : 2352 pixels

### Regroupement des classes

Les 7 classes originales du dataset ont été regroupées en 3 classes afin de réduire le déséquilibre des données :

| Classe | Description                                     |
| ------ | ----------------------------------------------- |
| 0      | Nevus mélanocytaire (nv)                        |
| 1      | Kératose bénigne (bkl)                          |
| 2      | Mélanome + Carcinome basocellulaire (mel + bcc) |

Distribution finale :

* Classe 0 : 6705 échantillons
* Classe 1 : 1099 échantillons
* Classe 2 : 1627 échantillons

---

## Prétraitement des données

Les principales étapes de préparation des données sont :

* Nettoyage des données
* Vérification des valeurs manquantes
* Réorganisation des classes
* Encodage des labels
* Normalisation des pixels
* Séparation Train / Validation / Test

Afin de limiter l’impact du déséquilibre des classes, une stratégie de pondération des classes (Class Weights) a été utilisée lors de l’entraînement.

---

## Modèles de Machine Learning

Trois algorithmes ont été évalués :

* Logistic Regression
* Random Forest
* XGBoost

Les performances sont comparées à l’aide de plusieurs métriques :

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Le meilleur modèle est automatiquement sélectionné à partir des performances obtenues sur les données de validation.

---




## Technologies utilisées

* Python
* Scikit-Learn
* XGBoost
* Pandas
* NumPy
* FastAPI
* Streamlit
* MLflow
* Docker
* GitHub Actions
* DVC
* Pytest
* Ruff

---



## Suivi des expérimentations

MLflow est utilisé pour :

* Enregistrer les métriques
* Sauvegarder les modèles
* Comparer les performances
* Conserver l’historique des expériences

---

## API REST avec FastAPI

L’API expose plusieurs endpoints :

### Vérification de l’état du service

```bash
GET /health
```

### Prédiction

```bash
POST /predict
```

Entrée :

```json
{
  "pixels": [...],
  "normalize": true
}
```

Sortie :

```json
{
  "predicted_index": 0,
  "predicted_label": "0",
  "probabilities": {
    "0": 0.92,
    "1": 0.05,
    "2": 0.03
  }
}
```

---

## Interface utilisateur

Une interface Streamlit permet :

* Le chargement d’une image
* L’extraction automatique des pixels
* L’envoi des données à l’API
* L’affichage du résultat de classification

---

## Monitoring

Un système de monitoring a été implémenté afin de suivre le comportement du modèle en production.

Les indicateurs surveillés sont :

* Latence des prédictions
* Probabilité maximale
* Classe prédite
* Mean Absolute Drift


Chaque prédiction est enregistrée dans des fichiers journaux JSONL.

---

## Intégration Continue

GitHub Actions est utilisé pour automatiser :

* L’analyse statique du code avec Ruff
* L’entraînement automatique sur un mini dataset
* L’exécution des tests unitaires avec Pytest

Cette approche permet de garantir la qualité du code à chaque modification.

---

## Containerisation

Le projet est entièrement dockerisé afin de faciliter :

* Le déploiement
* La reproductibilité
* La portabilité

