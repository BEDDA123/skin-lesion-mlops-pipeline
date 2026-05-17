# Calibration Removal & Overfitting Prevention

## Summary
Completely removed calibration from the pipeline to prevent data leakage and increased regularization to reduce overfitting.

## Changes Made

### 1. src/train.py

#### Removed Calibration Import (Line 21)
```python
# REMOVED:
from src.calibration import apply_calibration
```

#### Removed Calibration Logic (Lines 157-165)
```python
# REMOVED:
# Calibration probabiliste
if params.get("calibration", {}).get("enabled", True) and hasattr(model, "predict_proba"):
    try:
        cal_method = params.get("calibration", {}).get("method", "sigmoid")
        model = apply_calibration(model, X_train, y_train, X_val, y_val, method=cal_method)
        logger.info(f"Calibration applied: {cal_method}")
    except Exception as e:
        logger.warning(f"Calibration failed: {e}")
```

#### Removed Calibration from MLflow Params (Line 151)
```python
# REMOVED:
"calibration_enabled": params.get("calibration", {}).get("enabled", True),
```

#### Updated Default Rare Class Boost (Line 81)
```python
# CHANGED:
"rare_class_boost": float(imb.get("rare_class_boost", 1.2)),  # from 1.5 to 1.2
```

### 2. params.yaml

#### Disabled Calibration (Lines 20-23)
```yaml
# Calibration probabiliste (DISABLED - removed to prevent data leakage)
calibration:
  enabled: false
  method: "sigmoid"  # sigmoid | isotonic
```

#### Reduced Class Weight Boost (Line 17)
```yaml
rare_class_boost: 1.2  # reduced from 1.5 to prevent overfitting
```

#### Increased Logistic Regression Regularization (Line 35)
```yaml
C: 0.5  # already increased from 0.1 to 0.5
```

#### Increased Random Forest Regularization (Lines 39-40)
```yaml
max_depth: 10  # reduced from 15 to 10
min_samples_leaf: 4  # increased from 2 to 4
```

#### Increased XGBoost Regularization (Lines 49-50)
```yaml
reg_alpha: 0.5  # increased from 0.1 to 0.5
reg_lambda: 2.0  # increased from 1.0 to 2.0
```

## Why This Fixes the Problem

### Calibration Data Leakage
**Problem**: `CalibratedClassifierCV(cv=5)` was fitting on validation set, then evaluating on same data → 100% validation accuracy.

**Solution**: Completely removed calibration. Models now use raw probabilities without post-processing.

### Overfitting Prevention
**Problem**: Models were overfitting to training data, especially minority classes.

**Solutions**:
1. **Reduced class weight boost**: 1.2x instead of 1.5x → less emphasis on minority classes
2. **Stronger regularization**:
   - Logistic: C=0.5 (stronger L2)
   - RF: max_depth=10, min_samples_leaf=4 (simpler trees)
   - XGBoost: reg_alpha=0.5, reg_lambda=2.0 (stronger L1+L2)

## Expected Results

### Before
- Validation accuracy: ~100% (unrealistic)
- Test accuracy: Much lower
- Gap indicates data leakage/overfitting

### After
- Validation accuracy: Realistic (70-85%)
- Test accuracy: Similar to validation
- Gap should be minimal (<5%)

## Verification

Run training:
```bash
python -m src.train --force-prepare
```

Check logs for:
1. No "Calibration applied" message
2. Validation accuracy NOT 100%
3. Validation and test metrics closer together
4. Weight ratio reduced (expected ~4-6x instead of ~10-15x)

## Notes

- The `src/calibration.py` file is kept but not used
- Can be re-enabled later if needed for probability calibration
- Should be done on a separate calibration set, not validation set
- For now, raw model probabilities are used directly
