"""
Django settings for AEWIS MLOps Dashboard.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent  # Main AEWIS project root

SECRET_KEY = 'aewis-dashboard-dev-key-change-in-production'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'monitoring',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
            ],
        },
    },
]

STATIC_URL = '/static/'

# Path to AEWIS data files
AEWIS_PROJECT_ROOT = str(PROJECT_ROOT)
AEWIS_LOGS_DIR = os.path.join(AEWIS_PROJECT_ROOT, 'logs')
AEWIS_DATA_DIR = os.path.join(AEWIS_PROJECT_ROOT, 'data', 'processed')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
