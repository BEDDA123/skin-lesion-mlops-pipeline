"""
MLflow utilities for comprehensive model tracking and management.
Provides reusable functions for logging parameters, metrics, artifacts, and model registry operations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path


import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = None) -> None:
  
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set: {experiment_name}")


def log_hyperparameters(params: dict, model_name: str, model_params: dict) -> None:
   
    # Global parameters
    mlflow.log_params({
        "model": model_name,
        "n_classes": params.get("data", {}).get("n_classes", 3),
        "random_seed": params.get("data", {}).get("random_seed", 42),
        "test_size": params.get("data", {}).get("test_size", 0.15),
        "val_size": params.get("data", {}).get("val_size", 0.15),
        "imbalance_strategy": params.get("imbalance", {}).get("strategy", "balanced_minority"),
        "rare_class_boost": params.get("imbalance", {}).get("rare_class_boost", 1.2),
        "rare_classes": str(params.get("imbalance", {}).get("rare_classes", [])),
    })
    
    # Model-specific parameters
    mlflow.log_params(model_params)
    
    logger.info(f"Logged {len(model_params)} model-specific hyperparameters")


def log_metrics(metrics: dict, prefix: str = "") -> None:
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"{prefix}{metric_name}", metric_value)
    
    logger.info(f"Logged {len(metrics)} metrics with prefix '{prefix}'")


def log_model_artifacts(model, model_name: str, artifacts_dir: Path) -> None:
    
    # Log sklearn model
    mlflow.sklearn.log_model(model, artifact_path="model")
    
    # Log additional artifacts
    model_path = artifacts_dir / f"{model_name}.joblib"
    if model_path.exists():
        mlflow.log_artifact(str(model_path), artifact_path="models")
    
    logger.info(f"Logged model artifacts for {model_name}")


def log_evaluation_artifacts(
    reports_dir: Path,
    model_name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_encoder=None
) -> None:
    
    # Log confusion matrix
    cm_path = reports_dir / f"confusion_{model_name}_test.png"
    if cm_path.exists():
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
    
    # Log classification report
    report_path = reports_dir / f"classification_report_{model_name}.json"
    if report_path.exists():
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
    
    # Log diagnostic
    diagnostic_path = reports_dir / f"diagnostic_{model_name}.json"
    if diagnostic_path.exists():
        mlflow.log_artifact(str(diagnostic_path), artifact_path="diagnostics")
    
    # Generate and log ROC curves (for multi-class)
    if y_proba is not None and len(y_proba.shape) == 2:
        roc_path = reports_dir / f"roc_curve_{model_name}.png"
        generate_roc_curve(y_test, y_proba, label_encoder, roc_path)
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path), artifact_path="evaluation")
    
    logger.info(f"Logged evaluation artifacts for {model_name}")


def generate_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_encoder=None,
    output_path: Path = None
) -> None:
    
    n_classes = y_proba.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    # Get class names
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        # Binary classification for each class
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {output_path}")
    else:
        plt.close()


def log_class_weights(class_weights: dict, artifacts_dir: Path) -> None:
    """
    Log class weights as artifact.
    
    Args:
        class_weights: Dictionary of class weights
        artifacts_dir: Directory to save class weights
    """
    weights_path = artifacts_dir / "class_weights.json"
    if weights_path.exists():
        mlflow.log_artifact(str(weights_path), artifact_path="artifacts")
        logger.info("Logged class weights artifact")


def register_best_model(
    run_id: str,
    model_name: str,
    metric_name: str = "test_f1_macro",
    stage: str = "Production"
) -> str:
    """
    Register the best model in MLflow Model Registry.
    
    Args:
        run_id: MLflow run ID
        model_name: Name to register the model
        metric_name: Metric used to select the best model
        stage: Stage to promote the model to (Staging, Production, Archival)
    
    Returns:
        Registered model version
    """
    model_uri = f"runs:/{run_id}/model"
    
    # Register model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Transition to specified stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage
    )
    
    logger.info(f"Registered model {model_name} version {model_version.version} to stage {stage}")
    
    return model_version.version


def compare_runs(experiment_name: str, metric_name: str = "test_f1_macro") -> dict:
   
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) == 0:
        raise ValueError(f"No runs found in experiment {experiment_name}")
    
    # Find best run by metric
    best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmax()]
    
    return {
        "run_id": best_run["run_id"],
        "model": best_run["params.model"],
        "metric_value": best_run[f"metrics.{metric_name}"],
        "all_metrics": {k: v for k, v in best_run.items() if k.startswith("metrics.")}
    }


def log_training_summary(
    n_train: int,
    n_val: int,
    n_test: int,
    training_time: float,
    class_distribution: dict
) -> None:
   
    mlflow.log_params({
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "training_time_seconds": training_time,
    })
    
    # Log class distribution as JSON
    class_dist_path = Path("class_distribution.json")
    with open(class_dist_path, "w") as f:
        json.dump(class_distribution, f, indent=2)
    mlflow.log_artifact(str(class_dist_path), artifact_path="artifacts")
    class_dist_path.unlink()
    
    logger.info("Logged training summary")
