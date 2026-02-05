#!/bin/bash

# Deployment Script for Early Warning System API

set -e

echo "========================================="
echo "Early Warning System - Deployment Script"
echo "========================================="

# Configuration
IMAGE_NAME="early-warning-api"
IMAGE_TAG="${1:-latest}"
REGISTRY="${2:-localhost}"

echo ""
echo "Step 1: Build Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo ""
echo "Step 2: Tag image for registry..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "Step 3: Push to registry (optional)..."
read -p "Push to registry? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    echo "✓ Image pushed to registry"
fi

echo ""
echo "Step 4: Deploy with Docker Compose..."
docker-compose down
docker-compose up -d

echo ""
echo "Step 5: Wait for services to be healthy..."
sleep 10

echo ""
echo "Step 6: Check service health..."
curl -f http://localhost:8000/health || echo "WARNING: API health check failed"

echo ""
echo "========================================="
echo "Deployment Summary"
echo "========================================="
echo "✓ API: http://localhost:8000"
echo "✓ API Docs: http://localhost:8000/docs"
echo "✓ MLflow: http://localhost:5000"
echo "✓ Grafana: http://localhost:3000 (admin/admin)"
echo "✓ Prometheus: http://localhost:9090"
echo ""
echo "View logs: docker-compose logs -f api"
echo "========================================="
