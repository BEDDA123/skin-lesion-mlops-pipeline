from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend to prevent tkinter errors

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.class_weights import (
    compute_balanced_class_weights,
    sample_weights_from_class_dict,
    analyze_class_weights,
)
from src.config import ROOT, is_fast_train, load_params, resolve_raw_csv
from src.diagnostic import run_full_diagnostic
from src.evaluation import (
    classification_report_dict,
    compute_metrics,
    save_confusion_matrix_png,
)
from src.ingestion import load_csv, log_dataset_stats, split_features_labels, validate_dataset
from src.mlflow_utils import (
    log_class_weights,
    log_evaluation_artifacts,
    log_hyperparameters,
    log_metrics,
    log_model_artifacts,
    log_training_summary,
    register_best_model,
    setup_mlflow_experiment,
)
from src.preprocessing import (
    apply_imbalance_method,
    encode_labels,
    load_processed,
    normalize_pixels,
    save_processed,
    stratified_splits,
)
from src.training_utils import maybe_subsample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")


def run_prepare(force: bool = False) -> Path:
    params = load_params()
    raw_path = resolve_raw_csv(params)
    processed_dir = ROOT / params["data"]["processed_dir"]
    if (processed_dir / "dataset.npz").is_file() and not force:
        logger.info("Données déjà préparées: %s", processed_dir)
        return processed_dir

    df = load_csv(raw_path)
    summary = validate_dataset(df)
    log_dataset_stats(summary)

    X, y = split_features_labels(df)
    X = normalize_pixels(X, max_val=params["preprocess"]["normalize_max"])
    y_enc, le = encode_labels(y)

    seed = int(params["data"]["random_seed"])
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(
        X,
        y_enc,
        test_size=float(params["data"]["test_size"]),
        val_size=float(params["data"]["val_size"]),
        random_state=seed,
    )

    X_tr, y_tr = apply_imbalance_method(X_train, y_train)

    imb = params.get("imbalance", {})
    meta = {
        "summary": summary,
        "imbalance_method": "none",
        "imbalance": {
            "strategy": imb.get("strategy", "balanced_minority"),
            "rare_class_boost": float(imb.get("rare_class_boost", 1.2)),
            "rare_classes": list(imb.get("rare_classes", [0, 2])),
        },
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    save_processed(processed_dir, X_tr, X_val, X_test, y_tr, y_val, y_test, le, meta)

    legacy_sw = processed_dir / "train_sample_weight.npy"
    if legacy_sw.is_file():
        legacy_sw.unlink()
    legacy_cw = processed_dir / "class_weight.json"
    if legacy_cw.is_file():
        legacy_cw.unlink()

    return processed_dir


def train_all(processed_dir: Path | None = None) -> dict:
    params = load_params()
    processed_dir = processed_dir or (ROOT / params["data"]["processed_dir"])
    bundle = load_processed(processed_dir)
    le = bundle["label_encoder"]
    n_classes = len(le.classes_)

    X_train, y_train = bundle["X_train"], bundle["y_train"]
    X_val, y_val = bundle["X_val"], bundle["y_val"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]

    sample_rows = params["train"].get("sample_rows")
    if is_fast_train():
        sample_rows = sample_rows or 3000

    X_train, y_train, X_val, y_val = maybe_subsample(
        X_train, y_train, X_val, y_val, sample_rows, int(params["data"]["random_seed"])
    )

    n_features = int(X_train.shape[1])
    
    # Setup MLflow experiment
    setup_mlflow_experiment(
        experiment_name=params["mlflow"]["experiment_name"],
        tracking_uri=params["mlflow"]["tracking_uri"]
    )

    artifacts_dir = ROOT / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_rows: list[dict] = []
    best_name: str | None = None
    best_f1 = -1.0

    class_weights = _build_class_weights(y_train, params)
    logger.info(
        "Poids de classes (stratégie=%s): %s",
        params.get("imbalance", {}).get("strategy", "balanced_minority"),
        class_weights,
    )

    weight_analysis = analyze_class_weights(class_weights)
    logger.info(
        f"Analyse des poids: min={weight_analysis['min_weight']:.4f}, max={weight_analysis['max_weight']:.4f}, ratio={weight_analysis['weight_ratio']:.2f}x"
    )

    _save_class_weights_artifact(class_weights, artifacts_dir)

    model_names = ["logistic_regression", "random_forest", "xgboost"]
    run_ids = {}  # Store run IDs for model registration

    for model_name in model_names:
        with mlflow.start_run(run_name=model_name, nested=False) as run:
            run_id = run.info.run_id
            run_ids[model_name] = run_id
            
            # Log hyperparameters using new utility
            model_params = _get_model_params(model_name, params)
            log_hyperparameters(params, model_name, model_params)
            
            # Log params.yaml as artifact
            mlflow.log_artifact(ROOT / "params.yaml")

            # Training time tracking
            import time
            training_start = time.time()
            
            model = _fit_sklearn(model_name, X_train, y_train, X_val, y_val, params, class_weights)
            
            training_time = time.time() - training_start

            # Predictions
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
            # Save model artifacts
            if model_name == "logistic_regression":
                joblib.dump(model, artifacts_dir / "logreg.joblib")
            elif model_name == "random_forest":
                joblib.dump(model, artifacts_dir / "rf.joblib")
            elif model_name == "xgboost":
                joblib.dump(model, artifacts_dir / "xgb.joblib")
            
            # Log model artifacts using new utility
            log_model_artifacts(model, model_name, artifacts_dir)

            # Log metrics using new utility
            val_metrics = compute_metrics(y_val, val_pred)
            test_metrics = compute_metrics(y_test, test_pred)
            log_metrics(val_metrics, prefix="val_")
            log_metrics(test_metrics, prefix="test_")

            # Log training time
            mlflow.log_metric("training_time_seconds", training_time)

            # Generate and log evaluation artifacts
            cm_path = reports_dir / f"confusion_{model_name}_test.png"
            save_confusion_matrix_png(y_test, test_pred, cm_path, le)
            
            # Save classification report
            rep = classification_report_dict(y_test, test_pred, le)
            report_path = reports_dir / f"classification_report_{model_name}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
            
            # Log all evaluation artifacts (ROC curves, confusion matrix, etc.)
            log_evaluation_artifacts(
                reports_dir, model_name, y_test, test_pred, test_proba, le
            )

            # Diagnostic détaillé
            try:
                diagnostic = run_full_diagnostic(
                    model,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    X_train,
                    y_train,
                    label_encoder=le,
                    output_path=reports_dir / f"diagnostic_{model_name}.json",
                )
                logger.info(
                    f"Diagnostic for {model_name}: entropy_ratio={diagnostic['test_entropy'].get('entropy_ratio_mean', 'N/A'):.3f}"
                )
            except Exception as e:
                logger.warning(f"Diagnostic échoué pour {model_name}: {e}")

            # Log class weights
            log_class_weights(class_weights, artifacts_dir)

            results_rows.append(
                {
                    "model": model_name,
                    "val_f1_macro": val_metrics["f1_macro"],
                    "test_f1_macro": test_metrics["f1_macro"],
                    "test_accuracy": test_metrics["accuracy"],
                }
            )

            if test_metrics["f1_macro"] > best_f1:
                best_f1 = test_metrics["f1_macro"]
                best_name = model_name

    assert best_name is not None
    with open(artifacts_dir / "best_model.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": best_name, "val_f1_macro": best_f1}, f, indent=2)

    df_res = pd.DataFrame(results_rows)
    df_res.to_csv(reports_dir / "model_comparison.csv", index=False)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="summary"):
        mlflow.log_artifact(reports_dir / "model_comparison.csv")

    logger.info("Meilleur modèle (F1 macro validation): %s", best_name)
    
    # Register best model in MLflow Model Registry if enabled
    if params.get("mlflow", {}).get("enable_model_registry", False):
        try:
            best_run_id = run_ids.get(best_name)
            if best_run_id:
                model_registry_name = params.get("mlflow", {}).get("model_registry_name", "skin_lesion_classifier")
                register_best_model(
                    run_id=best_run_id,
                    model_name=model_registry_name,
                    metric_name="test_f1_macro",
                    stage="Production"
                )
                logger.info(f"Best model registered in MLflow Model Registry: {model_registry_name}")
        except Exception as e:
            logger.warning(f"Failed to register model in MLflow Model Registry: {e}")
    
    return {"best_model": best_name, "comparison_csv": str(reports_dir / "model_comparison.csv")}


def _build_class_weights(y_train: np.ndarray, params: dict) -> dict[int, float]:
    imb = params.get("imbalance", {})
    strategy = imb.get("strategy", "balanced_minority")

    if strategy == "balanced_minority":
        return compute_balanced_class_weights(
            y_train,
            rare_classes=imb.get("rare_classes"),
            rare_class_boost=float(imb.get("rare_class_boost", 1.5)),
        )
    else:
        return compute_balanced_class_weights(
            y_train,
            rare_classes=imb.get("rare_classes"),
            rare_class_boost=float(imb.get("rare_class_boost", 1.0)),
        )


def _fit_sklearn(
    name: str, X_train, y_train, X_val, y_val, params: dict, class_weights: dict[int, float]
):
    t = params["train"]
    sw = sample_weights_from_class_dict(y_train, class_weights)
    seed = int(params["data"]["random_seed"])

    if name == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            C=float(t["logistic"]["C"]),
            class_weight=class_weights,
            solver="saga",
            random_state=seed,
        )
        model.fit(X_train, y_train)
        return model
    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(t["rf"]["n_estimators"]),
            max_depth=t["rf"].get("max_depth", 15),
            min_samples_leaf=int(t["rf"].get("min_samples_leaf", 2)),
            min_samples_split=int(t["rf"].get("min_samples_split", 5)),
            class_weight=class_weights,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        return model
    if name == "xgboost":
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            n_estimators=int(t["xgb"]["n_estimators"]),
            max_depth=int(t["xgb"]["max_depth"]),
            learning_rate=float(t["xgb"]["learning_rate"]),
            subsample=float(t["xgb"]["subsample"]),
            colsample_bytree=float(t["xgb"]["colsample_bytree"]),
            reg_alpha=float(t["xgb"].get("reg_alpha", 0.1)),
            reg_lambda=float(t["xgb"].get("reg_lambda", 1.0)),
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        model.fit(X_train, y_train, sample_weight=sw)
        return model
    raise ValueError(name)


def _save_class_weights_artifact(class_weights: dict[int, float], artifacts_dir: Path) -> None:
    with open(artifacts_dir / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in class_weights.items()}, f, indent=2)


def _get_model_params(model_name: str, params: dict) -> dict:
    """Extract model-specific hyperparameters for MLflow logging."""
    t = params["train"]
    
    if model_name == "logistic_regression":
        return {
            "C": float(t["logistic"]["C"]),
            "max_iter": int(t["logistic"]["max_iter"]),
            "solver": "saga",
        }
    elif model_name == "random_forest":
        return {
            "n_estimators": int(t["rf"]["n_estimators"]),
            "max_depth": t["rf"].get("max_depth", 15),
            "min_samples_leaf": int(t["rf"].get("min_samples_leaf", 2)),
            "min_samples_split": int(t["rf"].get("min_samples_split", 5)),
        }
    elif model_name == "xgboost":
        return {
            "n_estimators": int(t["xgb"]["n_estimators"]),
            "max_depth": int(t["xgb"]["max_depth"]),
            "learning_rate": float(t["xgb"]["learning_rate"]),
            "subsample": float(t["xgb"]["subsample"]),
            "colsample_bytree": float(t["xgb"]["colsample_bytree"]),
            "reg_alpha": float(t["xgb"].get("reg_alpha", 0.1)),
            "reg_lambda": float(t["xgb"].get("reg_lambda", 1.0)),
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Préparation des données et entraînement des modèles."
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Exécute uniquement la préparation."
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Exécute uniquement l'entraînement."
    )
    parser.add_argument(
        "--force-prepare", action="store_true", help="Recalcule le preprocessing même si présent."
    )
    args = parser.parse_args()

    if not args.train_only:
        run_prepare(force=args.force_prepare)
    if not args.prepare_only:
        train_all()


if __name__ == "__main__":
    main()
