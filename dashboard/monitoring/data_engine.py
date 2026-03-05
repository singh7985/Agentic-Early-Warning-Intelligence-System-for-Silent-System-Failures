"""
Data engine: loads AEWIS MLOps data from logs and processed CSVs.
Runs src/mlops modules live to produce fresh drift / alert / perf data.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops import (
    DriftDetector,
    PerformanceLogger,
    AlertingSystem, AlertThresholds, LogAlertHandler,
)


def _project_path(*parts):
    return os.path.join(str(PROJECT_ROOT), *parts)


# ── cached singletons ─────────────────────────────────────────────
_cache: dict = {}


def _load_perf_logs() -> dict:
    """Load exported performance logs JSON."""
    fp = _project_path('logs', 'mlops_performance_logs.json')
    if not os.path.exists(fp):
        return {}
    with open(fp) as f:
        return json.load(f)


def _load_features():
    """Load train / test feature CSVs."""
    if 'train' not in _cache:
        _cache['train'] = pd.read_csv(_project_path('data', 'processed', 'train_features.csv'))
        _cache['test'] = pd.read_csv(_project_path('data', 'processed', 'test_features.csv'))
    return _cache['train'], _cache['test']


# ── public API consumed by views.py ───────────────────────────────

def get_overview() -> dict:
    """High-level KPIs."""
    logs = _load_perf_logs()
    summary = logs.get('summary', {})
    pred = summary.get('predictions', {})
    tok = summary.get('token_usage', {})
    err = summary.get('errors', {})
    lat = summary.get('latency', {})

    # aggregate avg latency from component summaries
    lat_vals = [v.get('avg_ms', 0) for v in lat.values()] if isinstance(lat, dict) else []

    return {
        'total_predictions': pred.get('total_predictions', 0),
        'avg_confidence': round(pred.get('avg_confidence', 0), 3),
        'total_tokens': tok.get('total_tokens', 0),
        'estimated_cost_usd': round(tok.get('estimated_total_cost_usd', 0), 4),
        'total_errors': err.get('total_errors', 0),
        'error_rate': round(err.get('error_rate', 0) * 100, 2),
        'avg_latency_ms': round(sum(lat_vals) / max(len(lat_vals), 1), 1),
    }


def get_latency_data() -> dict:
    """Per-component latency stats + raw timeseries."""
    logs = _load_perf_logs()
    summary = logs.get('summary', {}).get('latency', {})
    raw = logs.get('latencies', [])

    components = sorted(summary.keys()) if isinstance(summary, dict) else []
    by_component = {}
    for comp in components:
        s = summary[comp]
        by_component[comp] = {
            'avg': round(s.get('avg_ms', 0), 1),
            'p50': round(s.get('p50_ms', 0), 1),
            'p95': round(s.get('p95_ms', 0), 1),
            'p99': round(s.get('p99_ms', 0), 1),
            'max': round(s.get('max_ms', 0), 1),
        }

    # time-series for chart (index = observation #)
    ts = {}
    for entry in raw:
        comp = entry.get('component', 'unknown')
        ts.setdefault(comp, []).append(round(entry.get('latency_ms', 0), 1))

    return {'by_component': by_component, 'timeseries': ts}


def get_token_data() -> dict:
    """Token usage distribution."""
    logs = _load_perf_logs()
    raw = logs.get('token_usage', [])
    prompt = [t['prompt'] for t in raw]
    completion = [t['completion'] for t in raw]
    total = [t['total'] for t in raw]
    cost = [round(t['cost'] * 1000, 2) for t in raw]  # milli-dollars for readability

    summary = logs.get('summary', {}).get('token_usage', {})
    return {
        'prompt': prompt,
        'completion': completion,
        'total': total,
        'cost_millicents': cost,
        'summary': {
            'total_tokens': summary.get('total_tokens', 0),
            'avg_per_query': round(summary.get('avg_tokens_per_query', 0)),
            'total_cost_usd': round(summary.get('estimated_total_cost_usd', 0), 4),
        }
    }


def get_prediction_data() -> dict:
    """Prediction log with confidence, latency, errors."""
    logs = _load_perf_logs()
    preds = logs.get('predictions', [])
    errors = logs.get('errors', [])

    confidences = [p.get('confidence', 0) for p in preds]
    engine_ids = [p.get('engine_id', 0) for p in preds]
    predicted = [round(p.get('predicted_rul', 0), 1) for p in preds]
    actual = [p.get('actual_rul') for p in preds]
    total_lat = [round(p.get('total_latency_ms', 0), 1) for p in preds]

    return {
        'engine_ids': engine_ids,
        'predicted_rul': predicted,
        'actual_rul': actual,
        'confidences': confidences,
        'total_latency': total_lat,
        'errors': errors,
    }


def get_drift_data() -> dict:
    """Run live drift detection on train vs test features."""
    train, test = _load_features()
    numeric_cols = [c for c in train.columns
                    if train[c].dtype in ['float64', 'float32', 'int64']
                    and c not in ['engine_id', 'cycle', 'RUL']]
    features = numeric_cols[:8]

    detector = DriftDetector(threshold=0.05)
    detector.set_reference_data(train[features])

    n = len(test)
    batches = [
        ("Week 1 (stable)", test[features].iloc[:n // 3]),
        ("Week 2 (mid)",    test[features].iloc[n // 3: 2 * n // 3]),
    ]
    # Inject drift in week 3
    batch3 = test[features].iloc[2 * n // 3:].copy()
    for col in features[:3]:
        batch3[col] = batch3[col] + batch3[col].std() * 1.5
    batches.append(("Week 3 (drifted)", batch3))

    results = []
    for name, batch in batches:
        res = detector.detect_data_drift(batch, features)
        results.append({
            'batch': name,
            'drift_detected': res.drift_detected,
            'score': round(res.drift_score, 3),
            'severity': res.severity,
            'affected': len(res.affected_features),
            'total_features': len(features),
            'affected_names': res.affected_features,
        })

    return {'features': features, 'batches': results}


def get_alert_data() -> dict:
    """Run live alerting simulation."""
    thresholds = AlertThresholds(
        min_confidence=0.5,
        confidence_degradation_rate=0.05,
        error_rate_threshold=0.10,
        latency_threshold_ms=500,
        max_allowed_stale_days=30,
    )
    alerting = AlertingSystem(thresholds=thresholds)
    alerting.suppress_duplicate_alerts = False

    # Confidence degradation sequence
    conf_seq = [
        0.92, 0.91, 0.89, 0.88, 0.85, 0.82, 0.78, 0.72, 0.65, 0.55,
        0.50, 0.45, 0.42, 0.38, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20,
    ]
    for c in conf_seq:
        alerting.check_confidence_degradation(c)

    alerting.check_error_rate(2, 100)
    alerting.check_error_rate(15, 100)
    alerting.check_latency(150)
    alerting.check_latency(850)
    alerting.check_latency(2500)
    alerting.check_drift_detection('medium', 5)
    alerting.check_drift_detection('high', 8)
    alerting.check_model_staleness(45)

    summary = alerting.get_alert_summary()
    alerts_list = []
    for a in alerting.alerts:
        alerts_list.append({
            'type': a.alert_type.value,
            'severity': a.severity.value,
            'message': a.message,
            'timestamp': a.timestamp.isoformat(),
            'acknowledged': a.acknowledged,
        })

    return {
        'summary': summary,
        'alerts': alerts_list,
        'confidence_history': conf_seq,
        'threshold': thresholds.min_confidence,
    }


def get_experiment_comparison() -> dict:
    """Static comparison from NB07/NB08 results."""
    return {
        'systems': ['ML-Only', 'ML + RAG', 'ML + RAG + Agents'],
        'mae': [14.2, 11.0, 7.8],
        'rmse': [17.8, 13.9, 9.9],
        'r_squared': [0.94, 0.96, 0.98],
        'f1_score': [0.99, 0.99, 1.00],
        'lead_time': [15, 18, 20],
    }
