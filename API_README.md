# Early Warning System API

Production-ready API for predicting remaining useful life (RUL) of turbofan engines using agentic AI.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Access API docs
open http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Automated Deployment

```bash
chmod +x deploy.sh
./deploy.sh latest localhost
```

## API Endpoints

### POST /predict
Predict remaining useful life for an engine.

**Request**:
```json
{
  "sensor_data": {
    "engine_id": 1,
    "cycle": 150,
    "sensor_1": 518.67,
    ...
  },
  "use_rag": true,
  "use_agents": true
}
```

**Response**:
```json
{
  "predicted_rul": 87.5,
  "confidence": 0.89,
  "risk_level": "medium",
  "explanation": "...",
  "recommendations": [...]
}
```

### POST /explain
Get detailed explanation for a prediction.

### GET /health
Check system health and component status.

### GET /metrics
Get performance metrics and statistics.

### POST /drift
Detect data drift in sensor readings.

## Python Client

```python
from src.api.client import EarlyWarningClient

client = EarlyWarningClient("http://localhost:8000")

# Make prediction
result = client.predict(sensor_data, use_agents=True)
print(f"RUL: {result['predicted_rul']} cycles")

# Get explanation
explanation = client.explain(engine_id=1, cycle=150, predicted_rul=87.5)

# Check health
health = client.health()
```

## Cloud Deployment

### Google Cloud Run

```bash
gcloud run deploy early-warning-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker build -t early-warning-api .
docker tag early-warning-api:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/early-warning-api:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/early-warning-api:latest

# Register task definition and create service
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
aws ecs create-service --cluster default --service-name early-warning --task-definition early-warning-api
```

## Monitoring

- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Architecture

```
┌─────────────────┐
│   Nginx Proxy   │  :80, :443
└────────┬────────┘
         │
┌────────▼────────┐
│   FastAPI App   │  :8000
└────────┬────────┘
         │
    ┌────┴────┬────────────┬─────────────┐
    │         │            │             │
┌───▼──┐  ┌──▼──────┐  ┌─▼────────┐  ┌─▼──────────┐
│MLflow│  │Postgres │  │Prometheus│  │  Grafana   │
└──────┘  └─────────┘  └──────────┘  └────────────┘
 :5000      :5432        :9090         :3000
```

## Environment Variables

```bash
ENV=production
MLFLOW_TRACKING_URI=http://mlflow:5000
LOG_LEVEL=INFO
```

## Performance

- **Latency**: 100-200ms per prediction
- **Throughput**: ~50 req/sec per worker
- **Concurrency**: 320 concurrent requests (4 workers × 80)
- **Memory**: ~500MB per worker

## License

MIT License - see LICENSE file for details.
