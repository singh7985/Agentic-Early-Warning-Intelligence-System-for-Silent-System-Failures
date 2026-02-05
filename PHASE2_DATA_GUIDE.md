# PHASE 2 — Data Ingestion & Understanding (Days 4–7)

## Overview

This phase covers comprehensive data collection, preprocessing, and exploratory analysis for the EWIS project.

**Objectives:**
1. ✅ Download and parse NASA C-MAPSS time-series dataset
2. ✅ Visualize sensor degradation patterns
3. ✅ Create Remaining Useful Life (RUL) labels
4. ✅ Split data with proper train/val/test stratification
5. ✅ Download and parse LogHub system logs
6. ✅ Create incident narratives from logs
7. ✅ Generate synthetic maintenance reports
8. ✅ Store clean, normalized text corpus

---

## 1. Time-Series Data: NASA C-MAPSS

### Dataset Overview

**C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) is a publicly available turbofan engine degradation dataset from NASA.

- **Source:** https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- **Engines:** 100–260 per dataset
- **Sensors:** 21 multivariate sensor readings
- **Operational Settings:** 3 features (pressure, temperature, etc.)
- **Cycles per Engine:** 130–360+ operational cycles

### Four Datasets (FD001–FD004)

| Dataset | Engines | Condition | Size |
|---------|---------|-----------|------|
| FD001 | 100 | Normal operation | ~32 MB |
| FD002 | 260 | Various operational conditions | ~120 MB |
| FD003 | 100 | Normal + induced faults | ~35 MB |
| FD004 | 248 | Various + induced faults | ~140 MB |

### Loading C-MAPSS Data

#### Option 1: Automatic Download (Kaggle API)

```bash
# Install kaggle CLI
pip install kaggle

# Configure credentials
# 1. Download from https://www.kaggle.com/settings/account
# 2. Place at ~/.kaggle/kaggle.json
# 3. chmod 600 ~/.kaggle/kaggle.json

# Download dataset
python scripts/download_cmapss.py
```

#### Option 2: Manual Download

```bash
# Go to https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
# Download and extract to: data/raw/CMAPSS/
```

### Python API

```python
from src.ingestion.cmapss_loader import CMAPSSDataLoader, prepare_cmapss_data

# Load dataset
loader = CMAPSSDataLoader("./data/raw/CMAPSS")
train_df, test_df, rul_test = loader.load_dataset('FD001')

# Or use convenience function
data = prepare_cmapss_data(
    dataset_name='FD001',
    test_engines_ratio=0.2,
    normalize=True
)

print(f"Train: {len(data['train'])} rows")
print(f"Val: {len(data['val'])} rows")
print(f"Test: {len(data['test'])} rows")
```

### Data Fields

```
- engine_id: Unique engine identifier (1–100)
- cycle: Operating cycle (1–max_cycle)
- op_setting_1/2/3: Operational settings
- sensor_1 to sensor_21: Multivariate sensor readings
- RUL: Remaining Useful Life (calculated)
```

### Key Metrics

- **Total training records:** 20,631 (FD001)
- **Engines in training:** 100
- **Test engines:** 100 (separate set)
- **Sensors monitored:** 21
- **Avg lifespan:** ~180–200 cycles
- **Failure definition:** Last recorded cycle for each engine

---

## 2. Text Data: LogHub

### LogHub Overview

**LogHub** is a collection of public log datasets from real systems. Useful for:
- Understanding system failure patterns
- Extracting incident narratives
- Building RAG knowledge base

**Available Datasets:**
- **HDFS**: Hadoop Distributed File System (11M logs, 55 MB)
- **BGL**: Blue Gene/L supercomputer (4.7M logs, 700 MB)
- **OpenStack**: Cloud infrastructure (207k logs, 300 MB)
- **Android**: Mobile OS (1.5M logs, large)

### Downloading LogHub

```bash
# Download HDFS and BGL logs
python scripts/download_cmapss.py --loghub-only

# Or all datasets
python scripts/download_cmapss.py --all
```

### Log Parsing Pipeline

```python
from src.ingestion.log_parser import LogParser, IncidentGrouper, SyntheticReportGenerator

# Parse logs
parser = LogParser()
logs_df = parser.parse_log_file('data/raw/LogHub/HDFS_2k.log')

# Group into incidents
grouper = IncidentGrouper(window_size=100)
incidents = grouper.group_by_error_bursts(logs_df)

# Generate narratives and reports
for incident in incidents:
    narrative = grouper.create_incident_narrative(incident)
    report = SyntheticReportGenerator.generate_report(incident, logs_df)
```

### Log Parsing Features

The `LogParser` class:
- Extracts structured fields: timestamp, log level, source, message
- Normalizes messages (removes IPs, PIDs, UUIDs)
- Handles various log formats
- Supports multi-line logs

The `IncidentGrouper` class:
- Groups consecutive error logs into incidents
- Creates human-readable narratives
- Identifies error patterns

The `SyntheticReportGenerator` class:
- Generates maintenance reports from incidents
- Classifies incident types (connection, memory, disk, timeout)
- Provides actionable recommendations

---

## 3. Feature Engineering

### Time-Series Feature Engineering

For the time-series data, we provide comprehensive feature engineering:

```python
from src.features.engineering import TimeSeriesFeatureEngineer, ChangePointDetector

engineer = TimeSeriesFeatureEngineer(
    window_sizes=[5, 10, 20],
    ewma_spans=[5, 10, 20]
)

# Add various feature types
df = engineer.add_rolling_statistics(df, sensor_cols)
df = engineer.add_ewma_features(df, sensor_cols)
df = engineer.add_difference_features(df, sensor_cols)
df = engineer.add_fourier_features(df)
df = engineer.add_trend_features(df, sensor_cols)
df = engineer.add_statistical_features(df, sensor_cols)

# Detect change points (degradation onset)
detector = ChangePointDetector()
onsets = detector.detect_degradation_onset(df, sensor_cols, method='pelt')
```

### Features Created

| Category | Features |
|----------|----------|
| **Rolling Stats** | mean, std, min, max (multiple window sizes) |
| **EWMA** | Exponential moving averages (multiple spans) |
| **Differences** | Delta values (multiple lags) |
| **Fourier** | sin/cos pairs for cyclical patterns |
| **Trend** | Linear regression slope over window |
| **Statistics** | Skewness, kurtosis |

### Change Point Detection

Two algorithms for detecting degradation onset:
1. **PELT** (Pruned Exact Linear Time): Fast, accurate
2. **Binary Segmentation**: Flexible, slower

---

## 4. Data Organization

### Directory Structure

```
data/
├── raw/
│   ├── CMAPSS/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   ├── RUL_FD001.txt
│   │   ├── ... (FD002–FD004)
│   └── LogHub/
│       ├── HDFS_2k.log
│       └── BGL_2k.log
│
└── processed/
    ├── train_FD001.csv
    ├── val_FD001.csv
    ├── test_FD001.csv
    └── text_corpus/
        ├── cleaned_logs.csv
        ├── incidents.json
        └── incident_narratives.txt
```

### Output Files

**Time-Series:**
- `train_FD001.csv` — Training set (80% of engines)
- `val_FD001.csv` — Validation set (20% of engines)
- `test_FD001.csv` — Test set (separate 100 engines)

**Text:**
- `cleaned_logs.csv` — Parsed and normalized logs
- `incidents.json` — Grouped incidents with metadata
- `incident_narratives.txt` — Human-readable incident reports

---

## 5. Exploratory Data Analysis (EDA)

### Notebook: `notebooks/01_eda_cmapss_loghub.ipynb`

**Sections:**
1. Download and parse C-MAPSS dataset
2. Visualize sensor degradation patterns
3. Create RUL labels
4. Split data by engine
5. Download LogHub datasets
6. Normalize and parse logs
7. Convert logs to incident narratives
8. Store clean text corpus

### Key Visualizations

1. **Sensor Degradation Patterns** — Line plots showing sensor trends over cycles
2. **Sensor Correlation Matrix** — Heatmap of inter-sensor correlations
3. **RUL Distribution** — Histogram and per-engine lifespan
4. **Incident Summary** — Error burst frequencies and types

---

## 6. Data Quality & Preprocessing

### Time-Series Preprocessing

- **Normalization** (Z-score): Μ = 0, σ = 1
- **Missing Values** — Checked (NASA data is clean)
- **Outlier Detection** — Visual inspection via plots
- **Engine ID Grouping** — Prevent leakage across engines

### Text Preprocessing

- **Timestamp Removal** — Standardized timestamps
- **ID/IP Masking** — Replace with placeholders
- **Lowercase Normalization** — Consistent casing
- **Whitespace Cleaning** — Remove extra spaces
- **Template Extraction** — Group similar messages

---

## 7. Train/Val/Test Split Strategy

### Time-Series Split

To prevent **temporal leakage**:
- **Train:** 80% of engines, all their cycles
- **Val:** 20% of engines, all their cycles
- **Test:** Separate 100 engines (held-out completely)

### Why This Approach?

- Each engine follows a unique degradation trajectory
- We want to test on engines the model has never seen
- Data leakage within an engine (train vs val) is prevented
- Mimics real-world scenario: predicting failure on new engines

### Example (FD001)

```
Train: 80 engines × ~131 cycles = ~10,500 records
Val:   20 engines × ~131 cycles = ~2,600 records
Test:  100 engines × ~131 cycles = ~13,000 records
```

---

## 8. Next Steps (PHASE 3)

### Baseline 1: ML-Only (Days 7–9)

- Build RUL prediction models (XGBoost)
- Train anomaly detection (Isolation Forest)
- Implement change-point detection (PELT)
- Evaluate on test set

### Baseline 2: ML + RAG (Days 10–12)

- Set up FAISS vector database
- Embed logs and maintenance docs
- Implement LangChain retrieval chain
- Augment ML predictions with retrieved context

### Baseline 3: ML + RAG + Agents (Days 13–16)

- Build LangGraph agent orchestration
- Implement multi-agent workflow
- Add tool-calling and reflection
- Evaluate decision lead time

---

## 9. Key Hyperparameters

### Feature Engineering

```python
window_sizes = [5, 10, 20]  # For rolling statistics
ewma_spans = [5, 10, 20]    # For exponential moving averages
fourier_features = 5         # Number of sin/cos pairs
trend_window = 10            # Window for trend estimation
```

### Data Splitting

```python
test_engines_ratio = 0.2    # 20% of training engines for validation
random_seed = 42            # For reproducibility
```

### Log Parsing

```python
window_size = 100           # Max cycle gap for incident grouping
error_keywords = [          # Keywords indicating errors
    'error', 'exception', 'failed', 'failure', 'fatal'
]
```

---

## 10. Troubleshooting

### Issue: "Kaggle API credentials not found"

**Solution:**
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Move downloaded `kaggle.json` to `~/.kaggle/`
4. Run: `chmod 600 ~/.kaggle/kaggle.json`

### Issue: "CMAPSS files not found"

**Solution:**
1. Verify download completed: `ls -lh data/raw/CMAPSS/`
2. Should have 12 files (4 datasets × 3 files each)
3. If missing, re-run download script

### Issue: Memory error with large LogHub datasets

**Solution:**
1. Process logs in chunks
2. Filter to smaller date ranges
3. Use only HDFS instead of BGL (smaller)

---

## 11. References

- **NASA C-MAPSS:** https://data.nasa.gov/dataset/CMAPS
- **Kaggle Dataset:** https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- **LogHub:** https://github.com/logpai/loghub
- **Ruptures (change-point):** https://ruptures.readthedocs.io/

---

**Status:** ✅ PHASE 2 COMPLETE

**Date:** 2026-02-04

**Next Phase:** PHASE 3 — Baseline 1 Implementation (Days 7–9)
