"""
Ensembles avancés et fusion de modèles pour améliorer les performances.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class WeightedVotingEnsemble:
    """
    Ensemble pondéré où chaque modèle a un poids basé sur sa performance.
    Les poids sont calculés selon le F1 macro de validation.
    """
    
    def __init__(self):
        self.models = []
        self.weights = []
        self.is_fitted = False
    
    def add_model(self, model: Any, weight: float | None = None) -> None:
        """Ajoute un modèle à l'ensemble."""
        self.models.append(model)
        if weight is None:
            weight = 1.0 / (len(self.models))
        self.weights.append(float(weight))
    
    def fit_weights_from_validation(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Calcule les poids des modèles basés sur leur F1 macro en validation.
        """
        from sklearn.metrics import f1_score
        
        f1_scores = []
        for model in self.models:
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)
        
        # Normaliser les F1 scores comme poids
        f1_array = np.array(f1_scores)
        self.weights = f1_array / np.sum(f1_array)
        
        logger.info(f"Weights fitted from F1 scores: {dict(zip(range(len(self.weights)), self.weights))}")
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction via vote pondéré."""
        if not self.is_fitted or not self.weights:
            raise ValueError("Ensemble not fitted. Call fit_weights_from_validation first.")
        
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilités via moyenne pondérée des modèles."""
        probas_list = []
        
        for model in self.models:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                probas_list.append(proba)
            else:
                # Fallback : créer proba à partir de prédictions binaires
                pred = model.predict(X)
                n_classes = len(np.unique(pred))
                proba = np.zeros((len(X), n_classes))
                proba[np.arange(len(X)), pred] = 1.0
                probas_list.append(proba)
        
        # Moyenne pondérée
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
    """
    Crée un ensemble Stacking avec métaclassifieur.
    
    Args:
        estimators: liste de (name, model)
        final_estimator: métaclassifieur (défaut: LogisticRegression)
    """
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
    """
    Crée un ensemble Voting simple.
    
    Args:
        voting: 'soft' (probabilités) ou 'hard' (votes)
        weights: poids pour chaque modèle
    """
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
    Analyse les prédictions de tous les modèles de l'ensemble.
    Identifie les désaccords et les confiances.
    """
    from sklearn.metrics import f1_score, accuracy_score
    
    predictions = []
    probas = []
    scores = []
    
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            probas.append(y_proba)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        scores.append({"accuracy": acc, "f1_macro": f1})
    
    # Désaccords entre modèles
    predictions_array = np.array(predictions)
    disagreement_count = 0
    unanimous_correct = 0
    unanimous_wrong = 0
    
    for i in range(len(X_test)):
        preds = predictions_array[:, i]
        if len(np.unique(preds)) > 1:
            disagreement_count += 1
        else:
            is_correct = (preds[0] == y_test[i])
            if is_correct:
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


def create_class_specific_ensemble(
    base_models: dict[int, Any],
) -> Any:
    """
    Crée un ensemble où chaque classe a son propre classificateur optimisé.
    
    Usage:
        ensemble = create_class_specific_ensemble({
            3: xgb_for_class_3,
            5: xgb_for_class_5,
        })
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
            
            # Prédictions du modèle général
            proba = self.general_model.predict_proba(X)
            
            # Ajuster les probabilités pour les classes spéciales
            for class_id, model in self.base_models.items():
                if hasattr(model, "predict_proba"):
                    special_proba = model.predict_proba(X)
                    # Renforcer la classe spéciale
                    boost_factor = 1.5
                    proba[:, class_id] *= boost_factor
            
            # Renormaliser
            proba = proba / np.sum(proba, axis=1, keepdims=True)
            return proba
        
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)
    
    return ClassSpecificEnsemble(base_models)
