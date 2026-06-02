

from __future__ import annotations

import argparse
import logging

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.config import ROOT, load_params
from src.preprocessing import load_processed

logger = logging.getLogger(__name__)


def objective_logistic(trial):
    """Objective pour LogisticRegression."""
    # Hyperparamètres à tuner
    C = trial.suggest_float("C", 0.001, 10.0, log=True)
    max_iter = trial.suggest_categorical("max_iter", [1000, 3000, 5000, 10000])

    params = load_params()
    processed_dir = ROOT / params["data"]["processed_dir"]
    bundle = load_processed(processed_dir)

    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]

    # Construire class_weights
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
        class_weight=class_weights,
        solver="saga",
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
    """Objective pour RandomForest."""
    n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300, 500])
    max_depth = trial.suggest_categorical("max_depth", [5, 10, 15, 20, None])
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 5, 10])
    min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10])

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
    """Objective pour XGBoost."""
    n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300, 500])
    max_depth = trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.1, 2.0)

    params = load_params()
    processed_dir = ROOT / params["data"]["processed_dir"]
    bundle = load_processed(processed_dir)

    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]

    # Sample weights
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
        model.fit(X_train, y_train, sample_weight=sw)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        return f1
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 0.0


def run_tuning(model_name: str, n_trials: int = 100) -> None:
    """Lance le tuning pour un modèle donné."""
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Starting hyperparameter tuning for {model_name}...")

    if model_name == "logistic_regression":
        objective = objective_logistic
    elif model_name == "random_forest":
        objective = objective_rf
    elif model_name == "xgboost":
        objective = objective_xgb
    else:
        raise ValueError(f"Unknown model: {model_name}")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner()

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best F1 macro: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    # Save results
    results_path = ROOT / "reports" / f"tuning_{model_name}.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best F1 macro: {study.best_value:.4f}\n")
        f.write("Best params:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters with Optuna")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest", "xgboost"],
        default="xgboost",
        help="Model to tune",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials",
    )
    args = parser.parse_args()

    run_tuning(args.model, args.trials)
