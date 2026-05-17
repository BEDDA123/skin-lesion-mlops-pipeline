# Data Leakage & Overfitting Analysis

## Problem
Validation metrics are 100% but test performance is much lower, indicating potential data leakage or severe overfitting.

## Pipeline Analysis

### Current Data Flow
```
1. Load raw data
2. Normalize pixels (divide by 255) ← Applied BEFORE split
3. Encode labels (with remapping) ← Applied BEFORE split
4. Stratified split (train/val/test)
5. No resampling (apply_imbalance_method returns original)
6. Compute class weights on TRAIN only
7. Train models with class weights
8. Calibrate on validation set
9. Evaluate on validation and test
```

## Potential Issues Identified

### 1. **Normalization Timing** (LOW RISK)
- **Issue**: Normalization applied before split
- **Impact**: LOW - since it's just dividing by 255 (fixed constant), not data-dependent
- **Verdict**: Not the cause of 100% validation accuracy

### 2. **Calibration on Validation Set** (HIGH RISK)
- **Issue**: Calibration uses validation set with 5-fold CV
- **Impact**: HIGH - model sees validation data during calibration
- **Code location**: `src/calibration.py` line 87
- **Problem**: CalibratedClassifierCV with cv=5 fits on validation data
- **Verdict**: LIKELY CAUSE - model overfits to validation set during calibration

### 3. **Class Weight Strength** (MEDIUM RISK)
- **Issue**: 1.5x boost on minority classes might be too strong
- **Current**: Classes 0 and 2 get 1.5x boost
- **Impact**: MEDIUM - could cause overfitting to minority classes
- **Verdict**: Possible contributor

### 4. **Validation Set Size** (LOW RISK)
- **Current**: 15% of remaining after test split (~12.75% total)
- **With 8917 samples**: ~1137 validation samples
- **Verdict**: Adequate size, not the issue

### 5. **Dataset Quality** (UNKNOWN RISK)
- **Issue**: Possible duplicate samples in dataset
- **Impact**: HIGH if duplicates exist
- **Verdict**: Needs investigation

## Root Cause Analysis

### Most Likely Cause: Calibration Data Leakage
The calibration step uses `CalibratedClassifierCV(cv=5)` on the validation set. This means:
- The model is fitted on training data
- Then calibration fits 5 models on validation data
- Evaluation happens on the same validation data
- This creates data leakage between calibration and evaluation

### Secondary Cause: Class Overfitting
With class weights, the model might be:
- Overfitting to minority classes (0 and 2)
- Just predicting the majority class (1) to get high accuracy
- Not learning meaningful features

## Proposed Fixes

### Fix 1: Disable Calibration During Training
**File**: `params.yaml`
```yaml
calibration:
  enabled: false  # Disable during training
  method: "sigmoid"
```

**Rationale**: Calibration should be done AFTER model selection, not during training.

### Fix 2: Separate Calibration Set
**File**: `src/train.py`
Modify the split to create 4 sets: train/val/calibration/test

```python
# Current: 15% test, 15% val, 70% train
# Proposed: 15% test, 10% calibration, 10% val, 65% train
```

### Fix 3: Reduce Class Weight Boost
**File**: `params.yaml`
```yaml
imbalance:
  rare_class_boost: 1.2  # Reduce from 1.5 to 1.2
```

### Fix 4: Add Regularization
**File**: `params.yaml`
```yaml
train:
  logistic:
    C: 0.5  # Increase regularization (reduce from 0.1 to 0.5)
  rf:
    max_depth: 10  # Reduce from 15 to 10
    min_samples_leaf: 4  # Increase from 2 to 4
  xgb:
    reg_alpha: 0.5  # Increase from 0.1 to 0.5
    reg_lambda: 2.0  # Increase from 1.0 to 2.0
```

### Fix 5: Check for Duplicates
**File**: Create script `check_duplicates.py`
```python
import pandas as pd

df = pd.read_csv("data/raw/hmnist_filtered.csv")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Check for near-duplicates (same features, different labels)
pixel_cols = [col for col in df.columns if col != 'label']
features = df[pixel_cols]
duplicates_features = features.duplicated().sum()
print(f"Duplicate feature rows: {duplicates_features}")
```

### Fix 6: Add Cross-Validation
**File**: `src/train.py`
Add cross-validation on training set before final evaluation to get true performance estimate.

## Immediate Actions

### Priority 1: Disable Calibration
```yaml
# params.yaml
calibration:
  enabled: false
```

### Priority 2: Reduce Class Weights
```yaml
# params.yaml
imbalance:
  rare_class_boost: 1.2
```

### Priority 3: Increase Regularization
```yaml
# params.yaml
train:
  logistic:
    C: 0.5
  rf:
    max_depth: 10
    min_samples_leaf: 4
```

## Verification Steps

After applying fixes:
1. Run training: `python -m src.train --force-prepare`
2. Check that validation accuracy is NOT 100%
3. Check that validation and test metrics are closer
4. Check confusion matrix for balanced predictions
5. Verify model is not just predicting majority class

## Good Practices to Prevent Data Leakage

1. **Always split before preprocessing** (except for fixed transformations)
2. **Fit preprocessing on training set only**
3. **Apply same preprocessing to val/test using training parameters**
4. **Never use validation/test data for any training step**
5. **Calibration should be separate from model selection**
6. **Use cross-validation on training set for hyperparameter tuning**
7. **Keep test set completely separate until final evaluation**

## Recommended Pipeline

```
1. Load raw data
2. Split into train/val/test (stratified)
3. Fit preprocessing on TRAIN only (normalization, encoding)
4. Apply preprocessing to train/val/test
5. Compute class weights on TRAIN only
6. Train models on TRAIN with class weights
7. Tune hyperparameters using cross-validation on TRAIN
8. Select best model based on VAL metrics
9. Calibrate best model on VAL (or separate calibration set)
10. Final evaluation on TEST (never touched before)
```
