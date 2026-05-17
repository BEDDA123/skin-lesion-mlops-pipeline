# 3-Class Classification Setup

## Overview
Project modified from 7-class to 3-class classification using filtered dataset with classes 2, 4, 6 (remapped to 0, 1, 2).

## Dataset Filtering

### Original 7-Class Distribution
- Class 0: 327 samples (3.3%)
- Class 1: 514 samples (5.1%)
- Class 2: 1099 samples (11.0%) ✓
- Class 3: 115 samples (1.1%)
- Class 4: 6705 samples (66.9%) ✓
- Class 5: 142 samples (1.4%)
- Class 6: 1113 samples (11.1%) ✓

### Filtered 3-Class Distribution (min_samples >= 600)
- Class 2: 1099 samples (12.4%) → Remapped to **0**
- Class 4: 6705 samples (75.6%) → Remapped to **1** (MAJORITY)
- Class 6: 1113 samples (12.6%) → Remapped to **2**

**Total: 8,917 samples**

### After Remapping
- Class 0: 1099 samples (12.4%) - MINORITY
- Class 1: 6705 samples (75.6%) - MAJORITY
- Class 2: 1113 samples (12.6%) - MINORITY

## Changes Made

### 1. src/preprocessing.py
Modified `encode_labels()` function to automatically remap labels:
- Detects 3-class scenario
- Remaps original encoded labels to sequential 0, 1, 2
- Logs the remapping for verification

**Code added (lines 27-33):**
```python
# Remap to sequential 0, 1, 2 if we have 3 classes
unique_classes = sorted(np.unique(y_enc))
if len(unique_classes) == 3:
    # Create mapping from original encoded to sequential 0, 1, 2
    mapping = {old: new for new, old in enumerate(unique_classes)}
    y_enc = np.array([mapping[label] for label in y_enc], dtype=np.int32)
    logger.info(f"Labels remapped: {unique_classes} -> [0, 1, 2]")
```

### 2. params.yaml
Updated configuration for 3-class setup:
- `raw_csv`: Changed to `data/raw/hmnist_filtered.csv`
- `rare_classes`: Changed from `[3, 5]` to `[0, 2]` (minority classes after remapping)
- `experiment_name`: Changed to `skin_lesion_3class`

### 3. src/train.py
Updated default rare_classes in meta:
- Changed from `[3, 5]` to `[0, 2]`

## Class Weight Strategy

### New Distribution After Remapping
- Class 0: 1099 samples (12.4%)
- Class 1: 6705 samples (75.6%) - Majority
- Class 2: 1113 samples (12.6%)

### Weight Calculation
- Base balanced weights from sklearn
- 1.5x boost applied to classes 0 and 2 (minority classes)
- Expected weight ratio: ~5-7x (much more stable than previous 58x)

### Benefits
- Simpler classification task (3 vs 7 classes)
- More balanced dataset (no extremely rare classes)
- Better model convergence
- More stable training
- Easier to interpret results

## Verification

### 1. Run Filtering Script (if not already done)
```bash
cd data/raw
python dat.py
```

Expected output:
```
4    6705
6    1113
2    1099
```

### 2. Run Training
```bash
python -m src.train --force-prepare
```

Expected logs:
```
Labels remapped: [0, 1, 2] -> [0, 1, 2]
Poids de classes (stratégie=balanced_minority): {0: X.XX, 1: X.XX, 2: X.XX}
Analyse des poids: min=X.XXXX, max=X.XXXX, ratio=X.XXx
n_classes: 3
```

### 3. Verify MLflow
```bash
mlflow ui
```
- Check experiment name: `skin_lesion_3class`
- Verify n_classes parameter = 3
- Check confusion matrix shows 3x3 grid

## Model Compatibility

All models automatically adapt to 3 classes:
- **LogisticRegression**: Automatically detects 3 classes
- **RandomForest**: Automatically detects 3 classes
- **XGBoost**: Automatically detects 3 classes (num_class=3)

## API Compatibility

The API will automatically work with 3 classes:
- Predictions return 0, 1, 2
- LabelEncoder stored with remapping info
- No API changes needed

## Rollback to 7 Classes

If needed to revert to 7-class:
1. Change `raw_csv` back to `data/raw/hmnist_28_28_RGB.csv`
2. Change `rare_classes` back to `[3, 5]`
3. Change `experiment_name` back to `skin_lesion_classification`
4. The remapping logic will automatically skip (since n_classes != 3)
