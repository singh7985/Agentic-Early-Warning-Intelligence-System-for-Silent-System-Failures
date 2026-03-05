"""AEWIS Dashboard URL configuration."""
from django.urls import path, include

urlpatterns = [
    path('', include('monitoring.urls')),
]
