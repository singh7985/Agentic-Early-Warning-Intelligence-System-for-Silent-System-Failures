# Phase 4: Modeling Report

## Model Performance — Combined Test RMSE (NB03, FD001–FD004)

| Model | Test RMSE | Notes |
|-------|-----------|-------|
| **Random Forest** | 19.27 | |
| **XGBoost** | 19.13 | Selected model |
| **GradientBoosting** | 19.48 | |
| **LSTM** | 18.55 | Best test RMSE; early stopped at epoch 35, best at epoch 5 |
| **TCN** | 19.60 | Early stopped at epoch 55, best at epoch 25 |

## NB07 System Evaluation (FD001 only)

| Metric | ML-Only | ML+RAG | Full System |
|--------|---------|--------|-------------|
| MAE | 11.23 | 11.23 | 11.23 |
| RMSE | 15.86 | 15.86 | 15.88 |
| R² | 0.67 | 0.67 | 0.67 |

## Conclusion
The best performing model by combined test RMSE is **LSTM** (18.55), though **XGBoost** (19.13) is selected as the primary model for its interpretability and inference speed. In the NB07 system evaluation on FD001 only, the system achieves RMSE = 15.86.