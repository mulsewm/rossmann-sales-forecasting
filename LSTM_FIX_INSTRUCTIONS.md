# LSTM Model Training Fix Instructions

## Problem Identified
The model is producing all NaN predictions because training failed (final training loss is NaN). This is caused by:
- Learning rate too high (0.001) causing exploding gradients
- No gradient clipping to prevent NaN values during training

## Solution: Update Model Building Cell

### Step 1: Find the Model Building Cell
Look for the cell with `def build_efficient_lstm` function (around cell 18, "## 8. Build Lightweight LSTM Model")

### Step 2: Update the Optimizer
Change this line:
```python
optimizer = Adam(learning_rate=0.001)
```

To this:
```python
# Lower learning rate and add gradient clipping to prevent NaN during training
optimizer = Adam(
    learning_rate=0.0005,  # Reduced from 0.001 to prevent exploding gradients
    clipnorm=1.0  # Gradient clipping to prevent NaN/Inf during training
)
```

### Step 3: Ensure Data Validation Ran
Before retraining, make sure you've run the "Data Quality Validation" cell (before scaling) to clean any NaN values in the training data.

### Step 4: Retrain the Model
1. Go back to the "Data Quality Validation" cell (before scaling)
2. Run it to clean NaN values
3. Re-run the scaling cell
4. Re-run the model building cell (with updated optimizer)
5. Re-run the training cell
6. Then return to predictions

## Expected Results
After these changes:
- Training loss should decrease normally (not NaN)
- Model should produce valid predictions
- Metrics can be calculated successfully

## Alternative: Even More Conservative Settings
If the model still fails, try even more conservative settings:
```python
optimizer = Adam(
    learning_rate=0.0001,  # Even lower learning rate
    clipnorm=0.5  # Tighter gradient clipping
)
```

