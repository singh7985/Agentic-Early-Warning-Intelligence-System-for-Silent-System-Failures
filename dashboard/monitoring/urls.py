"""Monitoring app URL configuration."""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/overview/', views.api_overview, name='api_overview'),
    path('api/latency/', views.api_latency, name='api_latency'),
    path('api/tokens/', views.api_tokens, name='api_tokens'),
    path('api/drift/', views.api_drift, name='api_drift'),
    path('api/alerts/', views.api_alerts, name='api_alerts'),
    path('api/predictions/', views.api_predictions, name='api_predictions'),
    path('api/experiments/', views.api_experiments, name='api_experiments'),
]
