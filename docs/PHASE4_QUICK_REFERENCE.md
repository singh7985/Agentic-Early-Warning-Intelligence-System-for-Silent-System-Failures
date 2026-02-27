# PHASE 4 Quick Reference

Quick commands and code snippets for ML model training.

---

## 🚀 5-Minute Quickstart

### Train XGBoost Model

```python
from src.models.baseline_ml import XGBoostRULPredictor

# Initialize
model = XGBoostRULPredictor(n_estimators=200)

# Train
model.fit(X_train, y_train, X_val, y_val)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}")

# Save
model.save_model('models/xgboost_rul.json')
```

### Train LSTM Model

```python
from src.models.deep_learning import LSTMRULPredictor, DeepLearningTrainer

# Initialize
model = LSTMRULPredictor(input_size=50, hidden_size=64)
trainer = DeepLearningTrainer(model, learning_rate=0.001)

# Train
history = trainer.train(X_train, y_train, X_val, y_val, epochs=100)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.2f}")

# Save
trainer.save_model('models/lstm_rul.pt')
```

### Compare Models

```python
from src.models.model_selector import ModelSelector

selector = ModelSelector(primary_metric='rmse')

# Add results
selector.add_model_results("XGBoost", train_metrics, test_metrics, 0.015, 45.2)
selector.add_model_results("LSTM", train_metrics, test_metrics, 0.023, 180.5)

# Select best
best_model, scores = selector.select_best_model()
print(f"Best Model: {best_model}")

# Visualize
selector.plot_comparison(save_path='comparison.png')
```

---

## 📊 Evaluation Cheatsheet

### Calculate Metrics

```python
from src.models.evaluation import RULEvaluator

evaluator = RULEvaluator(model_name="MyModel")
metrics = evaluator.calculate_metrics(y_true, y_pred)

# Available metrics:
# - rmse: Root Mean Squared Error
# - mae: Mean Absolute Error
# - r2: R² Score
# - mape: Mean Absolute Percentage Error
# - max_error: Maximum error
# - median_error: Median error
# - std_error: Standard deviation of errors
# - early_prediction_pct: % underestimates
# - late_prediction_pct: % overestimates
```

### Visualize Predictions

```python
# 6-panel comprehensive plot
fig = evaluator.plot_predictions(
    y_true, y_pred,
    title="Model: Predicted vs Actual RUL",
    save_path="outputs/predictions.png"
)
```

### Plot Training History

```python
# For LSTM/TCN models
fig = evaluator.plot_training_history(
    train_losses=history['train_losses'],
    val_losses=history['val_losses']
)
```

---

## 🔬 MLflow Integration

### Basic Usage

```python
from src.models.mlflow_utils import MLflowTracker

tracker = MLflowTracker(experiment_name="RUL_Prediction")

with tracker.start_run(run_name="XGBoost_v1"):
    tracker.log_params({'n_estimators': 200, 'max_depth': 6})
    tracker.log_metrics({'rmse': 25.3, 'mae': 18.7})
    tracker.log_sklearn_model(model.model, "model")
```

### Convenience Functions

```python
from src.models.mlflow_utils import log_xgboost_model, log_deep_learning_model

# XGBoost
log_xgboost_model(tracker, model.model, params, metrics, feature_names, "XGBoost")

# LSTM/TCN
log_deep_learning_model(tracker, model, params, metrics, history, "LSTM")
```

### Start MLflow UI

```bash
cd /path/to/project
mlflow ui
# Open http://localhost:5000
```

---

## 🎯 Model Comparison

### Add Multiple Models

```python
selector = ModelSelector(primary_metric='rmse', lower_is_better=True)

# XGBoost
selector.add_model_results(
    model_name="XGBoost",
    train_metrics={'rmse': 20.1, 'mae': 15.2, 'r2': 0.90},
    test_metrics={'rmse': 25.3, 'mae': 18.7, 'r2': 0.85},
    inference_time_ms=0.015,
    training_time_seconds=45.2,
    model_size_mb=0.5
)

# Random Forest
selector.add_model_results("RandomForest", train_metrics, test_metrics, 0.05, 60.3, 2.0)

# LSTM
selector.add_model_results("LSTM", train_metrics, test_metrics, 0.023, 180.5, 0.1)
```

### Select Best Model

```python
# Default: 70% performance, 30% stability
best_model, scores = selector.select_best_model()

# Custom weights
best_model, scores = selector.select_best_model(
    stability_weight=0.4,
    performance_weight=0.6
)

# Filter by inference time
best_model, scores = selector.select_best_model(
    max_inference_time_ms=0.02  # Max 20ms
)
```

### Export Results

```python
# Get comparison table
df = selector.get_comparison_table()

# Export to CSV
selector.export_results('outputs/model_comparison.csv')

# Plot comparison
selector.plot_comparison(
    metrics=['rmse', 'mae', 'r2'],
    save_path='outputs/comparison.png'
)
```

---

## 🛠️ Common Tasks

### Load Data & Engineer Features

```python
from src.ingestion.cmapss_loader import CMAPSSDataLoader, prepare_cmapss_data
from src.features.pipeline import FeatureEngineeringPipeline

# Load
loader = CMAPSSDataLoader()
df_train, df_test = loader.load_dataset('FD001')
df_train_rul, df_test_rul, y_test = prepare_cmapss_data(df_train, df_test)

# Feature engineering
pipeline = FeatureEngineeringPipeline()
X_train_fe = pipeline.fit_transform(df_train_rul)
X_test_fe = pipeline.transform(df_test_rul)
```

### Train-Val Split

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, 
    df_train_rul['RUL'], 
    test_size=0.2, 
    random_state=42
)
```

### Save/Load Models

```python
# XGBoost (JSON)
model.save_model('models/xgboost_rul.json')
model.load_model('models/xgboost_rul.json')

# LSTM/TCN (PyTorch)
trainer.save_model('models/lstm_rul.pt')
trainer.load_model('models/lstm_rul.pt')
```

### Make Predictions

```python
# Single prediction
y_pred_single = model.predict(X_test[0:1])

# Batch predictions
y_pred = model.predict(X_test)
```

---

## ⚙️ Configuration Options

### XGBoost Parameters

```python
XGBoostRULPredictor(
    n_estimators=200,        # Number of trees
    max_depth=6,             # Tree depth
    learning_rate=0.1,       # Step size
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Column sampling
    random_state=42
)
```

### LSTM Parameters

```python
LSTMRULPredictor(
    input_size=50,           # Number of features
    hidden_size=64,          # LSTM hidden units
    num_layers=2,            # Stacked layers
    dropout=0.2,             # Dropout rate
    bidirectional=False      # Bidirectional LSTM
)
```

### TCN Parameters

```python
TCNRULPredictor(
    input_size=50,
    num_channels=[64, 64, 64],  # Channel sizes
    kernel_size=3,               # Conv kernel
    dropout=0.2
)
```

### Trainer Parameters

```python
trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,                  # Max epochs
    batch_size=128,              # Batch size
    early_stopping_patience=15   # Stop if no improvement
)
```

---

## 🐛 Troubleshooting Quick Fixes

### GPU Out of Memory

```python
# Reduce batch size
trainer.train(X_train, y_train, X_val, y_val, batch_size=64)

# Reduce model size
model = LSTMRULPredictor(hidden_size=32, num_layers=1)
```

### XGBoost Overfitting

```python
model = XGBoostRULPredictor(
    learning_rate=0.05,      # Reduce learning rate
    max_depth=4,             # Reduce complexity
    reg_alpha=1.0,           # L1 regularization
    reg_lambda=1.0           # L2 regularization
)
```

### LSTM Not Learning

```python
# Increase learning rate
trainer = DeepLearningTrainer(model, learning_rate=0.01)

# Normalize inputs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```

### MLflow Not Saving

```python
# Ensure context manager is used
with tracker.start_run(run_name="test"):
    tracker.log_params(params)
    tracker.log_metrics(metrics)
    # Complete all logging before exiting context
```

---

## 📁 File Locations

### Models

```
src/models/
├── __init__.py           # Package init
├── baseline_ml.py        # XGBoost, RF, GB
├── deep_learning.py      # LSTM, TCN
├── evaluation.py         # Metrics & plots
├── mlflow_utils.py       # MLflow tracking
└── model_selector.py     # Model comparison
```

### Notebooks

```
notebooks/
└── 03_ml_model_training.ipynb  # Complete workflow
```

### Outputs

```
outputs/
├── model_comparison.png        # Comparison dashboard
├── model_comparison.csv        # Metrics table
├── predictions.png             # 6-panel plot
└── phase4_final_summary.json   # Final results
```

### Saved Models

```
models/
├── xgboost_rul.json     # XGBoost model
├── lstm_rul.pt          # LSTM weights
└── tcn_rul.pt           # TCN weights
```

---

## 🔍 Inspection Commands

### Check GPU Availability

```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# GPU info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Check Model Parameters

```python
# XGBoost
params = model.get_params()
print(params)

# PyTorch
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### Check Data Shapes

```python
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")
```

### Verify No NaN

```python
import numpy as np
assert not np.isnan(X_train).any(), "NaN in X_train"
assert not np.isnan(y_train).any(), "NaN in y_train"
```

---

## 📚 Import Cheatsheet

```python
# Data & Features
from src.ingestion.cmapss_loader import CMAPSSDataLoader, prepare_cmapss_data
from src.features.pipeline import FeatureEngineeringPipeline

# Baseline Models
from src.models.baseline_ml import (
    XGBoostRULPredictor,
    RandomForestRULPredictor,
    GradientBoostingRULPredictor
)

# Deep Learning
from src.models.deep_learning import (
    LSTMRULPredictor,
    TCNRULPredictor,
    DeepLearningTrainer
)

# Evaluation
from src.models.evaluation import RULEvaluator

# MLflow
from src.models.mlflow_utils import (
    MLflowTracker,
    log_xgboost_model,
    log_deep_learning_model
)

# Model Selection
from src.models.model_selector import ModelSelector

# Config
from src.config import Config
from src.logging_config import setup_logging
```

---

## ⏱️ Time Estimates

| Task | Duration |
|------|----------|
| Train XGBoost | ~45 seconds |
| Train Random Forest | ~60 seconds |
| Train Gradient Boosting | ~50 seconds |
| Train LSTM (100 epochs) | ~3 minutes |
| Train TCN (100 epochs) | ~2.5 minutes |
| Evaluate model | <5 seconds |
| Generate 6-panel plot | ~10 seconds |
| Model comparison (5 models) | <30 seconds |
| Complete notebook (all cells) | ~10-15 minutes |

*Times are approximate for C-MAPSS FD001 dataset*

---

## 🎓 Next Steps

1. **Run the notebook:** `jupyter notebook notebooks/03_ml_model_training.ipynb`
2. **Read the guide:** `docs/PHASE4_ML_TRAINING_GUIDE.md`
3. **Experiment:** Try different hyperparameters
4. **Compare:** Train all 5 models and compare
5. **Deploy:** Move to PHASE 5 for RAG integration

---

**Quick Reference Version:** 1.1  
**Last Updated:** February 27, 2026
