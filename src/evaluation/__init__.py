"""
Evaluation Framework for Early Warning System

This module provides comprehensive evaluation tools for comparing:
1. ML only baseline
2. ML + RAG (retrieval-augmented)
3. ML + RAG + Agents (full system)

Metrics measured:
- RUL Prediction Error
- Early Warning Lead-Time
- Groundedness Score (explanation quality)
- False Alarm Rate
- Precision/Recall at different thresholds
"""

__version__ = "1.0.0"

from .metrics import MetricsCalculator
from .comparison import SystemComparison
from .ablation import AblationStudy
from .failure_analysis import FailureAnalyzer
from .evaluator import SystemEvaluator

__all__ = [
    'MetricsCalculator',
    'SystemComparison',
    'AblationStudy',
    'FailureAnalyzer',
    'SystemEvaluator',
]
