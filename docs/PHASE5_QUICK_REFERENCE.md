# PHASE 5 Quick Reference — Anomaly & Change-Point Detection

Quick commands and code snippets for anomaly detection and early warning systems.

---

## 🚀 5-Minute Quickstart

### Residual Anomaly Detection

```python
from src.anomaly import ResidualAnomalyDetector

# Initialize
detector = ResidualAnomalyDetector(method='zscore', threshold=3.0)

# Fit on training residuals
detector.fit(residuals_train)

# Detect anomalies
anomalies = detector.detect(residuals_test)
scores = detector.get_anomaly_scores(residuals_test)

# Visualize
fig = detector.plot_residuals(residuals_test, anomalies)
```

### Isolation Forest

```python
from src.anomaly import IsolationForestDetector

# Initialize
detector = IsolationForestDetector(contamination=0.1, n_estimators=100)

# Fit on training features
detector.fit(X_train)

# Detect anomalies
anomalies = detector.detect(X_test)
scores = detector.get_anomaly_scores(X_test)

# Visualize
fig = detector.plot_anomalies(X_test, anomalies)
```

### Change-Point Detection

```python
from src.anomaly import ChangePointDetector

# Initialize
detector = ChangePointDetector(method='cusum', threshold=3.0)

# Fit on baseline data
detector.fit(y_baseline)

# Detect change points
change_points = detector.detect(y_test)

# Visualize
fig = detector.plot_change_points(y_test, change_points)
```

### Degradation Labeling

```python
from src.anomaly import DegradationLabeler

# Initialize
labeler = DegradationLabeler(rul_threshold=100)

# Label degradation
df = labeler.label_degradation(
    rul_values=y_test,
    anomaly_flags=anomalies,
    anomaly_scores=scores,
    change_points=change_points
)

# Get periods
periods = labeler.get_degradation_periods(df)

# Visualize
fig = labeler.plot_degradation_labels(df)
```

### Early Warning System

```python
from src.anomaly import EarlyWarningSystem

# Initialize
system = EarlyWarningSystem(critical_rul=50, warning_rul=100)

# Generate warnings
warnings_df = system.generate_warnings(
    rul_values=y_test,
    anomaly_scores=anomaly_scores,
    degradation_scores=degradation_scores,
    change_points=change_points
)

# Calculate lead-time
stats = system.calculate_lead_time_statistics(warnings_df)

# Visualize
fig = system.plot_warnings(warnings_df)

# Export
system.export_warnings(warnings_df, 'warnings.csv')
```

---

## 📊 Method Cheatsheet

### Residual Detection Methods

| Method | Best For | Threshold | Notes |
|--------|----------|-----------|-------|
| **zscore** | Gaussian residuals | 3.0 std dev | Most common |
| **iqr** | Outlier robustness | 1.5 × IQR | Robust to extremes |
| **mad** | Robust statistics | 3.0 × MAD | Similar to IQR |
| **ewma** | Gradual shifts | 3.0 control limits | Detects trends |

```python
# Compare all methods
methods = ['zscore', 'iqr', 'mad', 'ewma']
for method in methods:
    detector = ResidualAnomalyDetector(method=method)
    detector.fit(residuals_train)
    anomalies = detector.detect(residuals_test)
    print(f"{method}: {np.sum(anomalies)} anomalies")
```

### Change-Point Methods

| Method | Best For | Threshold | Notes |
|--------|----------|-----------|-------|
| **cusum** | Mean shifts | 3.0 | Cumulative sum |
| **ewma** | Gradual changes | 3.0 | Weighted average |
| **bayesian** | Distribution changes | 0.95 (p-value) | Statistical test |
| **mann_kendall** | Monotonic trends | 3.0 (z-score) | Non-parametric |

```python
# Compare all methods
methods = ['cusum', 'ewma', 'bayesian', 'mann_kendall']
for method in methods:
    detector = ChangePointDetector(method=method)
    detector.fit(y_baseline)
    cps = detector.detect(y_test)
    print(f"{method}: {len(cps)} change points")
```

---

## ⚙️ Configuration Options

### Residual Detector

```python
ResidualAnomalyDetector(
    method='zscore',           # 'zscore', 'iqr', 'mad', 'ewma'
    threshold=3.0,             # Detection threshold
    window_size=50,            # Rolling window size
    contamination=0.1          # Expected anomaly rate
)
```

### Isolation Forest

```python
IsolationForestDetector(
    contamination=0.1,         # Expected anomaly proportion
    n_estimators=100,          # Number of trees
    max_samples='auto',        # Samples per tree
    max_features=1.0,          # Feature proportion
    random_state=42,           # Reproducibility
    normalize=True             # Standardize features
)
```

### Change-Point Detector

```python
ChangePointDetector(
    method='cusum',            # 'cusum', 'ewma', 'bayesian', 'mann_kendall'
    threshold=3.0,             # Detection threshold
    drift=0.5,                 # CUSUM drift parameter
    min_distance=10            # Min cycles between CPs
)
```

### Degradation Labeler

```python
DegradationLabeler(
    rul_threshold=100.0,       # RUL threshold for degradation
    anomaly_window=10,         # Window for anomaly rate
    anomaly_rate_threshold=0.3,  # Min anomaly rate
    change_point_proximity=20,   # Cycles after CP
    min_degradation_length=5     # Min degradation period
)
```

### Early Warning System

```python
EarlyWarningSystem(
    critical_rul=50.0,         # Critical RUL threshold
    warning_rul=100.0,         # Warning RUL threshold
    anomaly_threshold=0.5,     # Anomaly score threshold
    degradation_threshold=0.5, # Degradation score threshold
    min_warning_gap=10,        # Min cycles between warnings
    alert_levels={              # Custom alert thresholds
        'critical': 0.8,
        'high': 0.6,
        'medium': 0.4,
        'low': 0.2,
        'info': 0.0
    }
)
```

---

## 🔍 Common Workflows

### Complete Pipeline

```python
# 1. Train model (from PHASE 4)
from src.models.baseline_ml import XGBoostRULPredictor
model = XGBoostRULPredictor()
model.fit(X_train, y_train, X_val, y_val)

# 2. Get predictions and residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# 3. Anomaly detection (residual-based)
residual_detector = ResidualAnomalyDetector(method='zscore')
residual_detector.fit(residuals_train)
anomalies_residual = residual_detector.detect(residuals)
residual_scores = residual_detector.get_anomaly_scores(residuals)

# 4. Anomaly detection (multivariate)
iso_detector = IsolationForestDetector(contamination=0.1)
iso_detector.fit(X_train)
anomalies_iso = iso_detector.detect(X_test)

# 5. Change-point detection
cp_detector = ChangePointDetector(method='cusum')
cp_detector.fit(y_test[:100])  # Baseline
change_points = cp_detector.detect(y_test)

# 6. Degradation labeling
labeler = DegradationLabeler(rul_threshold=100)
degradation_df = labeler.label_degradation(
    rul_values=y_test,
    anomaly_flags=anomalies_residual,
    anomaly_scores=residual_scores,
    change_points=change_points
)

# 7. Early warnings
warning_system = EarlyWarningSystem(critical_rul=50, warning_rul=100)
warnings_df = warning_system.generate_warnings(
    rul_values=y_test,
    anomaly_scores=residual_scores,
    degradation_scores=degradation_df['degradation_score'].values,
    change_points=change_points
)

# 8. Calculate lead-time
lead_stats = warning_system.calculate_lead_time_statistics(warnings_df)
print(f"First warning lead-time: {lead_stats['first_warning_lead_time']:.0f} cycles")

# 9. Export results
warning_system.export_warnings(warnings_df, 'warnings.csv', format='csv')
labeler.plot_degradation_labels(degradation_df, save_path='degradation.png')
```

### Per-Engine Analysis

```python
# Analyze each engine separately
results = []

for engine_id in df_test['unit_number'].unique():
    engine_data = df_test[df_test['unit_number'] == engine_id]
    
    # Get RUL predictions
    y_engine = engine_data['RUL'].values
    X_engine = pipeline.transform(engine_data)
    y_pred_engine = model.predict(X_engine)
    
    # Detect anomalies
    residuals = y_engine - y_pred_engine
    anomalies = residual_detector.detect(residuals)
    
    # Generate warnings
    warnings_df = warning_system.generate_warnings(y_engine, anomaly_scores=residual_detector.get_anomaly_scores(residuals))
    
    # Statistics
    lead_stats = warning_system.calculate_lead_time_statistics(warnings_df)
    
    results.append({
        'engine_id': engine_id,
        'n_cycles': len(y_engine),
        'n_anomalies': np.sum(anomalies),
        'n_warnings': lead_stats['n_warnings'],
        'first_warning_lead_time': lead_stats['first_warning_lead_time']
    })

results_df = pd.DataFrame(results)
print(results_df)
```

---

## 📈 Inspection Commands

### Check Detector Statistics

```python
# Residual detector
stats = residual_detector.get_statistics()
print(f"Mean: {stats['residual_mean']:.4f}")
print(f"Std: {stats['residual_std']:.4f}")

# Isolation Forest
stats = iso_detector.get_statistics()
print(f"Contamination: {stats['contamination']:.2%}")
print(f"N Features: {stats['n_features']}")

# Change-point detector
stats = cp_detector.get_statistics()
print(f"Method: {stats['method']}")
print(f"Threshold: {stats['threshold']}")

# Degradation labeler
stats = labeler.get_statistics(degradation_df)
print(f"Degradation rate: {stats['degradation_rate']:.2%}")
print(f"N phases: {stats['n_phases']}")

# Early warning system
stats = warning_system.get_statistics(warnings_df)
print(f"N warnings: {stats['n_warnings']}")
print(f"Mean lead-time: {stats['mean_lead_time']:.0f}")
```

### Verify Data Quality

```python
# Check for NaN
assert not np.isnan(residuals).any(), "NaN in residuals"
assert not np.isnan(y_test).any(), "NaN in RUL values"

# Check shapes
print(f"Residuals: {residuals.shape}")
print(f"Features: {X_test.shape}")
print(f"RUL: {y_test.shape}")

# Check ranges
print(f"Residual range: [{residuals.min():.2f}, {residuals.max():.2f}]")
print(f"RUL range: [{y_test.min():.0f}, {y_test.max():.0f}]")
```

---

## 🐛 Troubleshooting

### Issue: Too Many Anomalies

**Problem:** Anomaly rate > 20%

**Solutions:**
```python
# Increase threshold
detector = ResidualAnomalyDetector(method='zscore', threshold=4.0)  # Instead of 3.0

# Use more robust method
detector = ResidualAnomalyDetector(method='mad')  # Instead of zscore

# Increase contamination (for Isolation Forest)
iso_detector = IsolationForestDetector(contamination=0.15)  # Instead of 0.1
```

### Issue: Too Few Change Points

**Problem:** No or very few change points detected

**Solutions:**
```python
# Lower threshold
detector = ChangePointDetector(method='cusum', threshold=2.0)  # Instead of 3.0

# Try different method
detector = ChangePointDetector(method='bayesian')  # More sensitive

# Reduce minimum distance
detector = ChangePointDetector(method='cusum', min_distance=5)  # Instead of 10
```

### Issue: No Degradation Detected

**Problem:** All samples labeled as normal

**Solutions:**
```python
# Increase RUL threshold
labeler = DegradationLabeler(rul_threshold=150)  # Instead of 100

# Lower anomaly rate threshold
labeler = DegradationLabeler(anomaly_rate_threshold=0.2)  # Instead of 0.3

# Reduce minimum length
labeler = DegradationLabeler(min_degradation_length=3)  # Instead of 5
```

### Issue: Too Many Warnings

**Problem:** Warning spam (too frequent)

**Solutions:**
```python
# Increase warning gap
system = EarlyWarningSystem(min_warning_gap=20)  # Instead of 10

# Raise thresholds
system = EarlyWarningSystem(
    critical_rul=30,  # Lower (only very critical)
    warning_rul=80    # Lower
)

# Increase alert level thresholds
system = EarlyWarningSystem(alert_levels={
    'critical': 0.9,  # Instead of 0.8
    'high': 0.7,      # Instead of 0.6
    # ...
})
```

---

## 💾 Save/Load Models

```python
# Save
residual_detector.save('models/residual_detector.pkl')
iso_detector.save('models/iso_detector.pkl')
cp_detector.save('models/cp_detector.pkl')
labeler.save('models/labeler.pkl')
warning_system.save('models/warning_system.pkl')

# Load
from src.anomaly import ResidualAnomalyDetector
detector = ResidualAnomalyDetector.load('models/residual_detector.pkl')
```

---

## 📁 File Locations

### Modules

```
src/anomaly/
├── __init__.py
├── residual_detector.py
├── isolation_forest_detector.py
├── change_point.py
├── degradation_labeler.py
└── early_warning.py
```

### Notebooks

```
notebooks/
└── 04_anomaly_detection.ipynb
```

### Outputs

```
outputs/
├── residual_anomalies.png
├── isolation_forest_anomalies.png
├── change_points.png
├── degradation_labels.png
├── early_warnings.png
├── warnings.csv
└── warnings.json
```

---

## 📚 Import Cheatsheet

```python
# All-in-one
from src.anomaly import (
    ResidualAnomalyDetector,
    IsolationForestDetector,
    ChangePointDetector,
    DegradationLabeler,
    EarlyWarningSystem
)

# Individual imports
from src.anomaly.residual_detector import ResidualAnomalyDetector
from src.anomaly.isolation_forest_detector import IsolationForestDetector
from src.anomaly.change_point import ChangePointDetector
from src.anomaly.degradation_labeler import DegradationLabeler
from src.anomaly.early_warning import EarlyWarningSystem
```

---

## ⏱️ Time Estimates

| Task | Duration |
|------|----------|
| Residual anomaly detection | ~5 seconds |
| Isolation Forest training | ~10 seconds |
| Isolation Forest detection | ~2 seconds |
| Change-point detection | ~3 seconds |
| Degradation labeling | ~2 seconds |
| Early warning generation | ~2 seconds |
| Complete pipeline (all steps) | ~30 seconds |
| Notebook (all cells) | ~2-3 minutes |

*Times for C-MAPSS FD001 test set (~13,000 samples)*

---

## 🎓 Next Steps

1. **Run the notebook:** `jupyter notebook notebooks/04_anomaly_detection.ipynb`
2. **Read the guide:** `docs/PHASE5_ANOMALY_DETECTION_GUIDE.md`
3. **Experiment:** Try different methods and thresholds
4. **Integrate:** Connect with PHASE 6 (RAG Pipeline)
5. **Deploy:** Move to production for real-time monitoring

---

**Quick Reference Version:** 1.0  
**Last Updated:** February 4, 2026
