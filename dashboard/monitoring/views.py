"""
Views for AEWIS MLOps Monitoring Dashboard.

- /              → Interactive HTML dashboard
- /api/overview  → JSON KPIs
- /api/latency   → JSON latency data
- /api/tokens    → JSON token usage
- /api/drift     → JSON drift detection results
- /api/alerts    → JSON alert data
- /api/predictions → JSON prediction data
- /api/experiments → JSON experiment comparison
"""
import json
from django.http import JsonResponse
from django.shortcuts import render

from . import data_engine


# ── HTML dashboard ────────────────────────────────────────────────

def dashboard(request):
    """Render the interactive monitoring dashboard."""
    overview = data_engine.get_overview()
    return render(request, 'monitoring/dashboard.html', {'overview': overview})


# ── JSON API endpoints ────────────────────────────────────────────

def api_overview(request):
    return JsonResponse(data_engine.get_overview())


def api_latency(request):
    return JsonResponse(data_engine.get_latency_data())


def api_tokens(request):
    return JsonResponse(data_engine.get_token_data())


def api_drift(request):
    return JsonResponse(data_engine.get_drift_data())


def api_alerts(request):
    return JsonResponse(data_engine.get_alert_data())


def api_predictions(request):
    return JsonResponse(data_engine.get_prediction_data())


def api_experiments(request):
    return JsonResponse(data_engine.get_experiment_comparison())
