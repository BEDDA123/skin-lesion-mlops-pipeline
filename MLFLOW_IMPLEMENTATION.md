# MLflow Tracking & Model Management Implementation

## Overview

Complete MLflow integration for tracking experiments, logging artifacts, comparing models, and managing the model lifecycle with the Model Registry.

## Architecture

### 1. MLflow Utilities Module (`src/mlflow_utils.py`)

Centralized module providing reusable MLflow functions:

- **`setup_mlflow_experiment()`**: Initialize experiment and tracking URI
- **`log_hyperparameters()`**: Log global and model-specific parameters
- **`log_metrics()`**: Log metrics with optional prefixes (val_, test_)
- **`log_model_artifacts()`**: Log trained models and artifacts
- **`log_evaluation_artifacts()`**: Log ROC curves, confusion matrices, reports
- **`generate_roc_curve()`**: Generate multi-class ROC curves
- **`log_class_weights()`**: Log class weight artifacts
- **`register_best_model()`**: Register model in MLflow Model Registry
- **`compare_runs()`**: Compare runs and find best model
- **`log_training_summary()`**: Log training metadata

### 2. Enhanced Training Pipeline (`src/train.py`)

Updated to use MLflow utilities:

- Automatic experiment setup
- Comprehensive hyperparameter logging
- Training time tracking
- ROC curve generation and logging
- Model comparison and best model selection
- Optional Model Registry integration

### 3. Configuration (`params.yaml`)

MLflow-specific configuration:

```yaml
mlflow:
  experiment_name: skin_lesion_3class
  tracking_uri: file:./mlruns
  enable_model_registry: false  # Set to true to enable
  model_registry_name: skin_lesion_classifier
```

## Features Implemented

### 1. Tracking Runs with MLflow

#### Hyperparameters Logged
- **Global parameters**: model, n_classes, random_seed, test_size, val_size
- **Imbalance parameters**: strategy, rare_class_boost, rare_classes
- **Model-specific parameters**: All hyperparameters for each model (C, max_iter, n_estimators, etc.)

#### Metrics Logged
- **Validation metrics**: val_accuracy, val_balanced_accuracy, val_precision_macro, val_recall_macro, val_f1_macro
- **Test metrics**: test_accuracy, test_balanced_accuracy, test_precision_macro, test_recall_macro, test_f1_macro
- **Training time**: training_time_seconds

### 2. Logging Artifacts

#### Models
- Trained models saved as joblib files
- Logged via `mlflow.sklearn.log_model()`
- Stored in `models/artifacts/`

#### Evaluation Artifacts
- **Confusion matrices**: PNG files for each model
- **Classification reports**: JSON files with per-class metrics
- **ROC curves**: Multi-class ROC curves with AUC scores
- **Diagnostics**: JSON files with entropy analysis
- **Model comparison**: CSV file comparing all models

#### Configuration Artifacts
- `params.yaml`: Full configuration used for training
- `class_weights.json`: Class weight distribution

### 3. Experiment Organization

#### Strategy
- **Single experiment**: `skin_lesion_3class`
- **Multiple runs**: One run per model (logistic_regression, random_forest, xgboost)
- **Summary run**: Additional run for model comparison artifact

#### Benefits
- Easy comparison between models in MLflow UI
- Consistent experiment tracking
- Simple filtering and querying

### 4. Model Comparison & Selection

#### Automatic Selection
- Models ranked by `test_f1_macro` on test set
- Best model saved to `models/artifacts/best_model.json`
- Comparison CSV saved to `reports/figures/model_comparison.csv`

#### Comparison Metrics
- Validation F1 macro
- Test F1 macro
- Test accuracy

### 5. MLflow Model Registry

#### Registration Process
- Best model automatically registered when `enable_model_registry: true`
- Registered with configurable name (`skin_lesion_classifier`)
- Promoted to "Production" stage automatically

#### Stages
- **Production**: Best model for deployment
- **Staging**: Can be used for pre-production testing
- **Archival**: Old models

## Usage

### Basic Training

```bash
# Train all models with MLflow tracking
python -m src.train
```

### Enable Model Registry

Update `params.yaml`:
```yaml
mlflow:
  enable_model_registry: true
```

Then train:
```bash
python -m src.train
```

### View MLflow UI

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### Compare Models

1. Open MLflow UI
2. Select experiment `skin_lesion_3class`
3. Compare runs in the "Compare" tab
4. View metrics, parameters, and artifacts side-by-side

### Query Best Model

```python
from src.mlflow_utils import compare_runs

best_run = compare_runs("skin_lesion_3class", "test_f1_macro")
print(f"Best model: {best_run['model']}")
print(f"F1 score: {best_run['metric_value']}")
```

## MLOps Best Practices

### 1. Separation of Concerns
- **MLflow utilities**: Reusable functions in dedicated module
- **Training logic**: Model training in train.py
- **Configuration**: All parameters in params.yaml

### 2. Reproducibility
- All hyperparameters logged automatically
- Configuration file logged as artifact
- Random seed fixed and logged
- Dataset metadata logged

### 3. Traceability
- Each run has unique ID
- Artifacts linked to specific runs
- Model lineage tracked
- Comparison metrics preserved

### 4. Deployment Readiness
- Model Registry for version control
- Automatic promotion to Production
- Clear model selection criteria
- Artifact organization for deployment

### 5. Monitoring
- Training time tracked
- Class weights logged
- Diagnostic metrics preserved
- ROC curves for probability calibration

## Integration with Docker

### Dockerfile Updates

```dockerfile
# MLflow tracking server
EXPOSE 5000

# Start MLflow UI with API
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
```

### docker-compose.yml

```yaml
services:
  mlflow:
    image: python:3.11
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
    command: mlflow ui --host 0.0.0.0 --port 5000
```

## Integration with FastAPI

### Loading Best Model

```python
from src.mlflow_utils import compare_runs
import mlflow

# Get best model from registry
best_run = compare_runs("skin_lesion_3class", "test_f1_macro")
model_uri = f"runs:/{best_run['run_id']}/model"
model = mlflow.sklearn.load_model(model_uri)
```

### Or from Model Registry

```python
# Load Production model from registry
model_uri = "models:/skin_lesion_classifier/Production"
model = mlflow.sklearn.load_model(model_uri)
```

## GitHub Integration

### CI/CD Pipeline

```yaml
name: Train and Register Model

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train models
        run: python -m src.train
      - name: Upload MLflow artifacts
        uses: actions/upload-artifact@v3
        with:
          name: mlflow-artifacts
          path: mlruns/
```

## Production Checklist

- [x] MLflow tracking configured
- [x] Hyperparameters logged
- [x] Metrics logged (validation and test)
- [x] Artifacts logged (models, ROC curves, confusion matrices)
- [x] Experiment organization implemented
- [x] Model comparison and selection
- [x] Model Registry integration
- [x] Configuration externalized
- [x] Documentation complete
- [ ] CI/CD pipeline integration
- [ ] Monitoring dashboard setup
- [ ] Model serving endpoint
- [ ] A/B testing framework

## Troubleshooting

### MLflow UI Not Starting
```bash
# Check tracking URI
echo $MLFLOW_TRACKING_URI

# Start with explicit host
mlflow ui --host 0.0.0.0 --port 5000
```

### Model Registry Errors
```bash
# Check MLflow backend store
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Artifact Logging Issues
```bash
# Check artifact permissions
ls -la models/artifacts/
ls -la reports/figures/
```

## Next Steps

1. **Enable Model Registry**: Set `enable_model_registry: true` in params.yaml
2. **Setup Remote Tracking**: Configure MLflow with remote backend (PostgreSQL/S3)
3. **Add Automated Retraining**: Schedule periodic retraining with GitHub Actions
4. **Implement Model Monitoring**: Track model performance in production
5. **Add A/B Testing**: Compare new models against production baseline
