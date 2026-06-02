"""
Ensembles avancés et fusion de modèles pour améliorer les performances.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class WeightedVotingEnsemble:
    

    def __init__(self):
        self.models = []
        self.weights = []
        self.is_fitted = False

    def add_model(self, model: Any, weight: float | None = None) -> None:
       
        self.models.append(model)
        if weight is None:
            weight = 1.0 / len(self.models)
        self.weights.append(float(weight))

    def fit_weights_from_validation(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        
        from sklearn.metrics import f1_score

        f1_scores = []
        for model in self.models:
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)

        f1_array = np.array(f1_scores)
        self.weights = f1_array / np.sum(f1_array)

        logger.info(
            f"Weights fitted from F1 scores: {dict(zip(range(len(self.weights)), self.weights))}"
        )
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or not self.weights:
            raise ValueError("Ensemble not fitted. Call fit_weights_from_validation first.")

        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas_list = []

        for model in self.models:
            if hasattr(model, "predict_proba"):
                probas_list.append(model.predict_proba(X))
            else:
                pred = model.predict(X)
                n_classes = len(np.unique(pred))
                proba = np.zeros((len(X), n_classes))
                proba[np.arange(len(X)), pred] = 1.0
                probas_list.append(proba)

        weighted_probas = np.average(
            np.array(probas_list),
            axis=0,
            weights=self.weights,
        )

        return weighted_probas


def create_stacking_ensemble(
    estimators: list[tuple[str, Any]],
    final_estimator: Any | None = None,
    cv: int = 5,
) -> StackingClassifier:
   
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=3000, multi_class="multinomial")

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
    )

    logger.info(f"Created StackingClassifier with {len(estimators)} base estimators")
    return stacking


def create_voting_ensemble(
    estimators: list[tuple[str, Any]],
    voting: str = "soft",
    weights: list[float] | None = None,
) -> VotingClassifier:
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1,
    )

    logger.info(f"Created VotingClassifier (voting={voting}) with {len(estimators)} estimators")
    return voting_clf


def ensemble_prediction_analysis(
    models: list[Any],
    model_names: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Analyse les prédictions de tous les modèles.
    """
    from sklearn.metrics import f1_score, accuracy_score

    predictions = []
    scores = []

    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)
        predictions.append(y_pred)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        scores.append({"accuracy": acc, "f1_macro": f1})

    predictions_array = np.array(predictions)

    disagreement_count = 0
    unanimous_correct = 0
    unanimous_wrong = 0

    for i in range(len(X_test)):
        preds = predictions_array[:, i]

        if len(np.unique(preds)) > 1:
            disagreement_count += 1
        else:
            if preds[0] == y_test[i]:
                unanimous_correct += 1
            else:
                unanimous_wrong += 1

    return {
        "model_scores": dict(zip(model_names, scores)),
        "disagreement_cases": int(disagreement_count),
        "unanimous_correct": int(unanimous_correct),
        "unanimous_wrong": int(unanimous_wrong),
        "disagreement_rate": float(disagreement_count / len(X_test)),
    }


def create_class_specific_ensemble(base_models: dict[int, Any]) -> Any:
    """
    Ensemble où chaque classe a un modèle spécialisé.
    """

    class ClassSpecificEnsemble:
        def __init__(self, base_models_dict):
            self.base_models = base_models_dict
            self.general_model = None

        def fit(self, X_train, y_train, general_model=None):
            self.general_model = general_model
            return self

        def predict_proba(self, X):
            if self.general_model is None:
                raise ValueError("Must fit with general_model first")

            proba = self.general_model.predict_proba(X)

            boost_factor = 1.5

            for class_id, model in self.base_models.items():
                if hasattr(model, "predict_proba"):
                    proba[:, class_id] *= boost_factor

            proba = proba / np.sum(proba, axis=1, keepdims=True)

            return proba

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    return ClassSpecificEnsemble(base_models)
