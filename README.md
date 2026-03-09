# Agentic Early-Warning Intelligence System for Silent System Failures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

A sophisticated agentic AI system that continuously analyzes time-series signals and unstructured logs to detect silent failure patterns before critical breakdowns occur. The system integrates ML-based RUL forecasting (XGBoost, Random Forest; with LSTM and TCN sequence models explored), Isolation Forest anomaly detection, RAG-based reasoning, and multi-agent orchestration to deliver explainable early warnings with actionable recommendations.

## 📋 Quick Links

- **[Research Paper](docs/RESEARCH_PAPER.md)** — Complete academic paper (8,500 words, 46 references)
- **[Final Report](docs/FINAL_REPORT.md)** — Comprehensive project report
- **[Project Status](#-project-status)** — Current phase and progress
- **[Getting Started](#-getting-started)** — Setup and usage
- **[Architecture](#-architecture)** — System design overview

---

## 🎯 Research Questions

| # | Research Question | Success Criteria | Result |
|---|-------------------|-----------------|--------|
| **RQ1** | Does agentic reasoning improve early-warning lead time? | ≥15% lead time improvement vs. baseline ML | Warning rate: 18%→23% (+28% relative). See NB07 for per-engine lead times. |
| **RQ2** | Does RAG improve interpretability and decision-maker trust? | Grounded, cited explanations | Groundedness: 0.00→0.13→0.39. No human evaluation conducted. |
| **RQ3** | When should the system abstain or escalate? | Calibrated escalation decisions | Computed from real ActionAgent decisions in NB07. |

---

## 🧪 Experimental Baselines

We compare three system variants to isolate the contribution of each component:

### **Baseline 1: ML-Only (Pure Predictive)**
- Time-series feature engineering (rolling stats, EWMA, health indicators)
- Random Forest, XGBoost, GradientBoosting for RUL prediction
- LSTM and TCN deep-learning models explored for sequence-based RUL
- Isolation Forest for multivariate anomaly detection
- NASA asymmetric scoring function for evaluation
- **No RAG, No agents**

### **Baseline 2: ML + RAG (Augmented ML)**
- Same time-series models as Baseline 1
- Vector DB (FAISS) with embedded logs, maintenance docs, failure reports
- Document chunking, sentence-transformer embeddings, similarity search
- KnowledgeBase retriever with citation tracking for contextual explanations
- **RAG enabled, No agents**

### **Baseline 3: ML + RAG + Agentic Reasoning (Full System)**
- Time-series models + RAG pipeline
- Multi-agent orchestration with 4 specialized agents:
  - **Monitoring Agent:** Continuous signal analysis, anomaly & drift detection
  - **Reasoning Agent:** Evidence synthesis, risk explanation, domain rules check
  - **Retrieval Agent:** Dynamic RAG with signal context, citation tracking
  - **Action Agent:** Intervention recommendations, confidence scoring, escalation logic
- Confidence thresholding and calibrated abstention
- **Full agentic system enabled**

---

## 📊 Evaluation Metrics

### Early-Warning Lead Time (Primary)
- **Detection Latency:** Cycles between first anomaly and system alert (target: minimize)
- **RUL Prediction RMSE:** Root mean squared error in cycles (target: minimize)
- **NASA Score:** Asymmetric scoring — penalizes late predictions more than early (target: minimize)
- **Lead Time Gain:** % improvement over Baseline 1 (target: >15%)

### Anomaly Detection Quality
- **Residual Threshold:** 95th percentile of validation residuals
- **Isolation Forest Threshold:** 5th percentile of training anomaly scores
- **Fusion Alerts:** Persistence-filtered warnings combining both detectors
- **Warning Rate:** % of test engines receiving pre-failure alerts

### RAG & Interpretability
- **Explanation Coherence** (structural groundedness scoring)
- **Citation Coverage** (% of explanations with KB references)
- **Groundedness Score** (composite of citations + pattern matches)

### Agentic Reasoning
- **Abstention Rate** (ReasoningAgent confidence thresholding)
- **Escalation Decisions** (ActionAgent + ReasoningAgent logic)
- **Computational Cost** (per-batch processing time)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Poetry (for dependency management)
- Kaggle account (to download NASA C-MAPSS dataset)
- GPU optional (CUDA or Apple MPS) for LSTM/TCN training

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures
   ```

2. **Set up Python environment with Poetry:**
   ```bash
   poetry install
   poetry shell
   ```

3. **Create `.env` file from template:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and paths
   ```

4. **Download NASA C-MAPSS dataset:**
   ```bash
   python scripts/download_cmapss.py
   ```

5. **Verify installation:**
   ```bash
   python -c "import src; print(src.__version__)"
   ```

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph INPUT["Data Ingestion"]
        A1["NASA C-MAPSS FD001\n21 sensors, 100 test engines"]
        A2["System Logs\nHDFS, BGL"]
    end

    subgraph FE["Feature Engineering"]
        B1["Rolling Statistics\nHealth Indicators\nSliding Windows"]
        B2["Feature Selection\nVariance, Correlation, Tree\nk = 20"]
        B3["Text Embeddings\nSentence-BERT, 384-dim"]
    end

    subgraph ML["Predictive Models"]
        C1["Random Forest\nXGBoost, GBR"]
        C2["LSTM, TCN\nSequence Models"]
        C3["RUL Prediction"]
    end

    subgraph AD["Anomaly Detection"]
        D1["Residual Analysis\nIsolation Forest\nChange-Point Detection"]
        D2["Degradation Labeling\nMulti-signal Fusion"]
        D3["Early Warning\n5-level Alert System"]
    end

    subgraph RAG["RAG Knowledge Base"]
        E1["Document Chunking\nFAISS Vector Store\nSemantic Retrieval"]
    end

    subgraph AGENTS["Multi-Agent Orchestration"]
        F1["Monitoring\nAgent"] --> F2["Retrieval\nAgent"]
        F2 --> F3["Reasoning\nAgent"]
        F3 --> F4["Action\nAgent"]
    end

    subgraph DEPLOY["Deployment"]
        G1["FastAPI\nREST API"]
        G2["MLflow\nTracking"]
        G3["Prometheus\nGrafana"]
    end

    A1 --> FE
    A2 -.-> B3
    B1 --> B2
    B3 --> E1
    B2 --> ML
    B2 --> AD
    C1 & C2 --> C3
    C2 --> D1
    D1 --> D2 --> D3
    C3 --> D2
    D3 --> E1
    ML --> AGENTS
    AD --> AGENTS
    RAG --> AGENTS
    AGENTS --> DEPLOY
```

### Compact System Architecture

```mermaid
flowchart LR
    A["Sensor Data\nC-MAPSS\n21 sensors"] --> B["Feature\nEngineering"]
    B --> C["ML Models\nRF, XGB, GBR\nLSTM, TCN"]
    B --> D["Anomaly\nDetection"]
    C & D --> E["Degradation\nLabeling &\nEarly Warning"]
    E --> F["RAG\nKnowledge\nBase"]
    C & D & F --> G["4-Agent\nOrchestrator"]
    G --> H["FastAPI\n+ MLOps"]
```

---

## 📁 Project Structure

```
agentic-ewis/
├── README.md                          # This file
├── pyproject.toml                     # Poetry dependencies
├── requirements.txt                   # Pip requirements (alternative)
├── .env.example                       # Environment variables template
├── project_config.json                # Project root configuration
│
├── src/                               # Source code (~15,800 lines, 51 modules)
│   ├── __init__.py                    # Package init, version 0.1.0
│   ├── config.py                      # Pydantic Settings configuration
│   ├── logging_config.py              # Structured logging setup
│   ├── data_loader_v2.py              # Data loading utilities
│   │
│   ├── ingestion/                     # Data loading & preprocessing
│   │   ├── cmapss_loader.py           #   NASA C-MAPSS dataset loader
│   │   └── log_parser.py             #   Log parsing & normalization
│   │
│   ├── features/                      # Feature engineering
│   │   ├── engineering.py             #   Rolling stats, EWMA, Fourier
│   │   ├── feature_selection.py       #   PCA, variance, correlation, importance
│   │   ├── health_indicators.py       #   Domain-specific health indices
│   │   ├── pipeline.py                #   Reproducible feature pipeline
│   │   └── sliding_windows.py         #   Fixed-size window generation
│   │
│   ├── models/                        # ML models
│   │   ├── baseline_ml.py             #   XGBoost, RF, GBR regressors
│   │   ├── deep_learning.py           #   LSTM, TCN architectures
│   │   ├── evaluation.py              #   Metrics & visualization
│   │   ├── mlflow_utils.py            #   MLflow integration
│   │   └── model_selector.py          #   Model comparison & selection
│   │
│   ├── anomaly/                       # Anomaly detection (~2,400 lines)
│   │   ├── change_point.py            #   CUSUM, EWMA, Bayesian, Mann-Kendall
│   │   ├── degradation_labeler.py     #   Silent degradation period labeling
│   │   ├── early_warning.py           #   Alert generation with lead-time
│   │   ├── isolation_forest_detector.py  # Multivariate anomaly detection
│   │   └── residual_detector.py       #   Z-score, IQR, MAD, EWMA residuals
│   │
│   ├── rag/                           # RAG pipeline (~2,300 lines)
│   │   ├── document_chunker.py        #   Multi-strategy chunking
│   │   ├── embedder.py                #   Sentence-transformer embeddings
│   │   ├── knowledge_base.py          #   Knowledge base builder
│   │   ├── retriever.py               #   Similarity search, re-ranking, citations
│   │   └── vector_store.py            #   FAISS vector DB with persistence
│   │
│   ├── agents/                        # Multi-agent system (~2,400 lines)
│   │   ├── monitoring_agent.py        #   ML inference, anomaly detection
│   │   ├── retrieval_agent.py         #   VectorDB historical context queries
│   │   ├── reasoning_agent.py         #   Risk explanation, evidence synthesis
│   │   ├── action_agent.py            #   Interventions, escalations
│   │   └── orchestrator.py            #   AgentOrchestrator sequential pipeline
│   │
│   ├── evaluation/                    # Evaluation framework (~1,800 lines)
│   │   ├── metrics.py                 #   RUL, warning, groundedness metrics
│   │   ├── comparison.py              #   3-baseline comparison
│   │   ├── ablation.py                #   Component ablation studies
│   │   ├── evaluator.py               #   High-level evaluation orchestrator
│   │   └── failure_analysis.py        #   Failure case root cause analysis
│   │
│   ├── mlops/                         # MLOps & monitoring (~1,300 lines)
│   │   ├── mlflow_tracker.py          #   Experiment tracking & model registry
│   │   ├── drift_detection.py         #   Data & prediction drift
│   │   ├── performance_logger.py      #   Token usage, latency logging
│   │   └── alerting.py                #   Performance monitoring, alert system
│   │
│   └── api/                           # REST API (~800 lines)
│       ├── main.py                    #   FastAPI app with all endpoints
│       └── client.py                  #   Python API client library
│
├── notebooks/                         # Jupyter notebooks (end-to-end pipeline)
│   ├── 01_eda_cmapss_loghub.ipynb     #   Phase 2: EDA & data ingestion
│   ├── 02_feature_engineering_baseline.ipynb  # Phase 3: Feature eng. + XGB/RF baselines
│   ├── 02_feature_engineering_pipeline.ipynb  # Phase 3: Production feature pipeline
│   ├── 03_ml_model_training.ipynb     #   Phase 4: LSTM, TCN, NASA Score, per-subset eval
│   ├── 04_anomaly_detection.ipynb     #   Phase 5: Residual + IF + fusion early warning
│   ├── 05_rag_pipeline.ipynb          #   Phase 6: FAISS vector store, retrieval, citations
│   ├── 06_agentic_architecture.ipynb  #   Phase 7: 4-agent orchestration, abstention
│   ├── 07_system_evaluation.ipynb     #   Phase 8: 3-baseline comparison, ablation study
│   └── 08_mlops_monitoring.ipynb      #   Phase 9: MLflow, drift, alerting
│
├── data/                              # Data storage
│   ├── raw/                           #   Raw NASA C-MAPSS (FD001–FD004)
│   │   └── CMAPSS/                    #     train/test/RUL .txt files
│   ├── interim/                       #   Combined parquet & CSV files
│   └── processed/                     #   NumPy arrays, features, thresholds
│       ├── X_train.npy, y_train.npy   #     Sliding-window sequences
│       ├── anomaly_thresholds.json    #     Detection thresholds
│       └── text_corpus/               #     Cleaned logs & incident narratives
│
├── models/                            # Trained model artifacts
│   ├── lstm_best.pth                  #   LSTM checkpoint (explored)
│   ├── tcn_best.pth                   #   TCN checkpoint (explored)
│   ├── xgb_model.joblib               #   XGBoost regressor
│   ├── xgboost_best.joblib            #   Tuned XGBoost (primary RUL model)
│   ├── xgb_rul_baseline.joblib        #   XGBoost RUL baseline
│   ├── rf_failure_baseline.joblib     #   Random Forest classifier
│   ├── if_anomaly_detector.joblib     #   Isolation Forest detector
│   ├── scaler.joblib                  #   StandardScaler
│   ├── metrics.json                   #   Model performance metrics
│   ├── features.json                  #   Selected feature list
│   └── checkpoints/                   #   Feature pipeline artifacts
│
├── reports/                           # Generated reports & figures
│   ├── eda/                           #   EDA summary CSVs & plots
│   └── figures/                       #   Modeling plots, feature importance
│
├── docs/                              # Phase documentation
│   ├── RESEARCH_PAPER.md              #   Complete academic paper (8,500 words)
│   ├── FINAL_REPORT.md                #   Project final report
│   ├── PHASE4_SUMMARY.md              #   ML training guide
│   ├── PHASE5_SUMMARY.md              #   Anomaly detection guide
│   ├── PHASE6_SUMMARY.md              #   RAG pipeline guide
│   ├── PHASE7_SUMMARY.md              #   Agentic architecture guide
│   ├── PHASE8_SUMMARY.md              #   Evaluation guide
│   └── PHASE12_SUMMARY.md             #   Final delivery
│
├── scripts/                           # Utility scripts
│   └── download_cmapss.py             #   Dataset downloader
│
├── Dockerfile                         # Multi-stage container build
├── docker-compose.yml                 # 7-service orchestration
├── nginx.conf                         # Reverse proxy configuration
├── prometheus.yml                     # Metrics collection config
├── cloudrun.yaml                      # GCP Cloud Run deployment
├── ecs-task-definition.json           # AWS ECS Fargate deployment
└── deploy.sh                          # Automated deployment script
```

---

## 📊 Comprehensive Results

> **Note:** All values below are computed from real C-MAPSS FD001 test data in [NB07](notebooks/07_system_evaluation.ipynb).  
> Run the notebook to see exact figures with all decimal places.

| Metric Category | Baseline 1 (ML-Only) | Baseline 2 (ML+RAG) | Baseline 3 (AEWIS) | Notes |
|----------------|---------------------|--------------------|--------------------|-------|
| **Predictive Performance** |
| RUL MAE (cycles) | 11.23 | 11.23 | 11.23 | XGBoost dominates; layers target detection, not MAE |
| RUL R² | 0.67 | 0.67 | 0.67 | Strong baseline leaves little room for MAE improvement |
| **Early Warning (RQ1)** |
| Warning Rate | 18% | 22% | 23% | +28% more engines warned vs ML-only |
| Lead Time | See NB07 | See NB07 | See NB07 | Computed from earliest warning per engine |
| **Explainability (RQ2)** |
| Groundedness | 0.00 | > 0 | > 0 | Real citations from KB retrieval |
| Explanation Coverage | N/A | Per-observation | Per-observation | ReasoningAgent produces text for every row |
| **Operational (RQ3)** |
| Escalation Decisions | N/A | N/A | Real agent logic | ActionAgent + ReasoningAgent escalation |
| Abstention Rate | 0% | 0% | Calibrated | ReasoningAgent confidence thresholding |

**Key Findings:**
- **RQ1:** Warning rate improved 18% → 23% (+28% relative). Per-engine lead times computed in NB07.  
  The value of agents is in **detection coverage**, not MAE — XGBoost with 112 features already saturates C-MAPSS FD001 prediction accuracy.
- **RQ2:** Groundedness: 0.00 (ML-only) → 0.13 (ML+RAG) → 0.39 (Full system).  
  No human evaluation was conducted; scores are structural (citation + pattern based).
- **RQ3:** Escalation decisions computed from real ActionAgent logic in NB07. No formal precision metric was computed.

---

## 🚀 Quick Start

### Local Development (Docker Compose)
```bash
# Clone and setup
git clone <repository-url>
cd Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures

# Start all services
docker-compose up -d

# Check API health
curl http://localhost:8000/health

# Access monitoring
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### Using the API
```python
from src.api.client import EarlyWarningClient

client = EarlyWarningClient("http://localhost:8000")

# Predict with full agentic system
response = client.predict(
    sensor_data=[100.0, 0.84, 518.67, ...],  # 21 sensors
    use_agents=True
)

print(f"RUL: {response['rul_prediction']:.2f} cycles")
print(f"Confidence: {response['confidence']:.2f}")
print(f"Alert Level: {response['alert_level']}")
```

### Running Notebooks
Execute the notebooks in sequence for the full pipeline:
```
01 → EDA & data ingestion
02 → Feature engineering (baseline or pipeline)
03 → ML model training (RF, XGB, GB, LSTM, TCN)
04 → Anomaly detection & early warning
05 → RAG pipeline integration
06 → Agentic architecture
07 → System evaluation & comparison
08 → MLOps monitoring
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md) | Complete academic paper (8,500 words) |
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | Project final report |
| [docs/PHASE6_SUMMARY.md](docs/PHASE6_SUMMARY.md) | RAG pipeline guide |
| [docs/PHASE7_SUMMARY.md](docs/PHASE7_SUMMARY.md) | Agentic architecture guide |
| [docs/PHASE8_SUMMARY.md](docs/PHASE8_SUMMARY.md) | Evaluation guide |
| [docs/PHASE12_SUMMARY.md](docs/PHASE12_SUMMARY.md) | Final delivery summary |

---

## 📊 Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation):**
- https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- Turbofan engine degradation simulation data
- 4 subsets: FD001 (1 fault, 1 condition) → FD004 (2 faults, 6 conditions)
- 21 sensors per engine, ~700 engines total
- Multi-sensor time-series with Remaining Useful Life (RUL) labels

---

## 🛠️ Development

### Running the API Server

**Local Development:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Docker Compose (Full Stack):**
```bash
# Start all services (API, MLflow, Postgres, Nginx, Prometheus, Grafana)
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

**Cloud Deployment:**
```bash
# Google Cloud Run
gcloud run deploy early-warning-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated

# AWS ECS Fargate
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition.json
```

### Code Quality
```bash
# Format code
black src/

# Sort imports
isort src/

# Lint
flake8 src/

# Type check
mypy src/
```

---

## 📈 Project Status

### ✅ Phase 1: Environment & Repo Setup (Days 1–3)
- [x] GitHub repo structure created
- [x] Python environment (Poetry + pip)
- [x] Core dependencies configured
- [x] .env, .gitignore, logging setup

### ✅ Phase 2: Data Ingestion & EDA (Days 4–7)
- [x] Download NASA C-MAPSS dataset (FD001–FD004)
- [x] Exploratory data analysis (sensor distributions, correlations, degradation patterns)
- [x] RUL label creation with clipping (CLIP_MAX=125)
- [x] Train/val/test splits by engine
- [x] Log parsing and incident narrative generation
- [x] **Notebook:** `01_eda_cmapss_loghub.ipynb`

### ✅ Phase 3: Feature Engineering (Days 8–14)
- [x] Rolling window statistics (mean, std, trend)
- [x] Health indicators and sensor drift features
- [x] Feature selection (variance, correlation, importance)
- [x] Sliding window sequences (SEQ_LEN=30)
- [x] Production feature pipeline with artifact saving
- [x] XGBoost + Random Forest baselines
- [x] **Notebooks:** `02_feature_engineering_baseline.ipynb`, `02_feature_engineering_pipeline.ipynb`

### ✅ Phase 4: ML Model Training (Days 15–22)
- [x] Random Forest, XGBoost, GradientBoosting regressors
- [x] LSTM deep learning model (60 max epochs, early stopped at 35, best checkpoint at epoch 5)
- [x] TCN deep learning model (60 max epochs, early stopped at 55, best checkpoint at epoch 25)
- [x] NASA asymmetric scoring function
- [x] Per-subset evaluation (FD001–FD004)
- [x] Model comparison with RMSE + NASA Score
- [x] **Notebook:** `03_ml_model_training.ipynb`

### ✅ Phase 5: Anomaly Detection & Early Warning (Days 23–27)
- [x] LSTM residual analysis (threshold from val 95th percentile)
- [x] Isolation Forest detector (threshold from train 5th percentile)
- [x] Fusion & early warning with persistence filter
- [x] 5-level Warning_Status (NORMAL → CRITICAL)
- [x] Anomaly threshold saving to JSON
- [x] **Notebook:** `04_anomaly_detection.ipynb`

### ✅ Phase 6: RAG Pipeline (Days 28–32)
- [x] FAISS vector DB setup with persistence
- [x] Document chunking (multi-strategy)
- [x] Sentence-transformer embeddings
- [x] Similarity search with re-ranking
- [x] Citation tracking for retrieved incidents
- [x] **Notebook:** `05_rag_pipeline.ipynb`

### ✅ Phase 7: Agentic Architecture (Days 33–38)
- [x] Monitoring Agent (signal analysis, drift detection)
- [x] Retrieval Agent (dynamic RAG with context)
- [x] Reasoning Agent (evidence synthesis, risk explanation)
- [x] Action Agent (interventions, escalation logic)
- [x] Orchestrator (collaborative agent coordination)
- [x] Confidence thresholding and abstention
- [x] **Notebook:** `06_agentic_architecture.ipynb`

### ✅ Phase 8: Evaluation & Analysis (Days 39–43)
- [x] 3-baseline comparison (ML-Only vs ML+RAG vs Full System)
- [x] Ablation study (7 configurations)
- [x] Failure case analysis and root cause analysis
- [x] Lead time analysis and per-engine warning evaluation
- [x] **Notebook:** `07_system_evaluation.ipynb`

### ✅ Phase 9: MLOps & Monitoring (Days 44–48)
- [x] MLflow experiment tracking & model registry
- [x] Drift detection (data & prediction)
- [x] Performance logging (token usage, latency)
- [x] Alerting system (confidence degradation)
- [x] **Notebook:** `08_mlops_monitoring.ipynb`

### ✅ Phase 10: API & Deployment (Days 49–54)
- [x] FastAPI backend (/predict, /explain, /health, /metrics, /drift)
- [x] Agent integration into API
- [x] Docker containerization (multi-stage build)
- [x] Docker Compose (6-service orchestration)
- [x] Cloud deployment configs (GCP Cloud Run, AWS ECS Fargate)
- [x] Python API client library
- [x] Monitoring stack (Prometheus, Grafana)

### ✅ Phase 11: Research Paper (Days 55–58)
- [x] Complete 8,500-word paper (46 references)
- [x] Tables with real NB07 evaluation results
- [x] Honest limitations and future work documented

### ✅ Phase 12: Final Delivery & Documentation (Days 59–60)
- [x] Final report with comprehensive results
- [x] All documentation updated and cross-referenced
- [x] Deployment configurations verified
- [x] Project handoff package complete

---

## 📚 References

- **Research Paper:** [docs/RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md) — Complete academic paper (8,500 words, 46 references)
- **NASA C-MAPSS Dataset:** https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- **FAISS:** https://github.com/facebookresearch/faiss
- **MLflow:** https://mlflow.org/
- **Evidently AI:** https://www.evidentlyai.com/
- **Phase Documentation:** [docs/](docs/) — Phase-by-phase summaries

---

## 📝 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 🙏 Acknowledgments

- NASA for the C-MAPSS dataset
- Sentence-Transformers and FAISS communities
- MLflow and Evidently communities
- PyTorch team for deep learning framework

---

**Last Updated:** 2026-03-09
**Status:** All 12 Phases Complete ✅

**Total Source Code:** ~15,800 lines across 51 modules in `src/`
**Notebooks:** 9 notebooks covering the full pipeline (EDA → Deployment)
