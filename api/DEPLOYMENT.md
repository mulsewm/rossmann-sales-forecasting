# API Deployment Guide

This guide covers deploying the Rossmann Sales Forecasting API.

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)
- Trained model files in the `models/` directory
- Processed data files (optional, for reference)

## Local Deployment

### Option 1: Direct Python

1. **Install dependencies:**
   ```bash
   cd api
   pip install -r requirements.txt
   pip install -r ../requirements.txt
   ```

2. **Ensure models are available:**
   ```bash
   # Models should be in ../models/
   ls ../models/*.pkl ../models/*.h5
   ```

3. **Start the API:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Option 2: Docker

1. **Build the Docker image:**
   ```bash
   cd api
   docker-compose build
   ```

2. **Start the container:**
   ```bash
   docker-compose up -d
   ```

3. **Check logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop the container:**
   ```bash
   docker-compose down
   ```

## Production Deployment

### Using Docker

1. **Build for production:**
   ```bash
   docker build -t rossmann-api:latest -f api/Dockerfile .
   ```

2. **Run container:**
   ```bash
   docker run -d \
     --name rossmann-api \
     -p 8000:8000 \
     -v $(pwd)/models:/app/models:ro \
     -v $(pwd)/data:/app/data:ro \
     -v $(pwd)/logs:/app/logs \
     rossmann-api:latest
   ```

### Using Docker Compose

1. **Deploy:**
   ```bash
   cd api
   docker-compose up -d
   ```

2. **Monitor:**
   ```bash
   docker-compose logs -f api
   ```

### Using Cloud Platforms

#### AWS (Elastic Beanstalk / ECS)

1. **Create Dockerfile** (already created)
2. **Deploy to ECS:**
   ```bash
   # Build and push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t rossmann-api .
   docker tag rossmann-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/rossmann-api:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/rossmann-api:latest
   ```

3. **Create ECS task definition** with:
   - Image: Your ECR image
   - Port: 8000
   - Environment variables as needed

#### Google Cloud Platform (Cloud Run)

1. **Build and deploy:**
   ```bash
   gcloud builds submit --tag gcr.io/<project-id>/rossmann-api
   gcloud run deploy rossmann-api \
     --image gcr.io/<project-id>/rossmann-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

#### Azure (Container Instances)

1. **Build and push to ACR:**
   ```bash
   az acr build --registry <registry-name> --image rossmann-api:latest .
   ```

2. **Deploy:**
   ```bash
   az container create \
     --resource-group <resource-group> \
     --name rossmann-api \
     --image <registry-name>.azurecr.io/rossmann-api:latest \
     --cpu 2 --memory 4 \
     --registry-login-server <registry-name>.azurecr.io \
     --ip-address Public \
     --ports 8000
   ```

## Environment Variables

Create a `.env` file or set environment variables:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false  # Set to true for development

# Model Configuration
MODEL_PATH=models/XGBoost_20251126_161803.pkl  # Default model
MODELS_DIR=models

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
```

## Load Balancing

For production, use a load balancer (nginx, AWS ALB, etc.):

### Nginx Configuration

```nginx
upstream rossmann_api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    server_name api.rossmann.com;

    location / {
        proxy_pass http://rossmann_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Health Checks

The API includes a health check endpoint:

```bash
curl http://localhost:8000/health
```

### Logging

Logs are written to:
- Console (stdout)
- `logs/` directory (if configured)

### Metrics (Optional)

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram
from fastapi import Request

request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'API request duration')
```

## Security

### API Keys (Recommended)

Add authentication middleware:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
```

### Rate Limiting

Add rate limiting:

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_sales(request: Request, ...):
    ...
```

## Scaling

### Horizontal Scaling

Run multiple instances behind a load balancer:

```bash
# Start multiple instances
uvicorn app:app --host 0.0.0.0 --port 8000 &
uvicorn app:app --host 0.0.0.0 --port 8001 &
uvicorn app:app --host 0.0.0.0 --port 8002 &
```

### Vertical Scaling

Increase resources:
- CPU: More workers
- Memory: Larger instances
- GPU: For LSTM models (if using GPU)

## Troubleshooting

### Model Not Loading

1. Check model files exist:
   ```bash
   ls -la models/
   ```

2. Check model path in logs:
   ```bash
   docker-compose logs api | grep "Model loaded"
   ```

3. Verify model format:
   ```python
   import joblib
   model = joblib.load("models/XGBoost_20251126_161803.pkl")
   ```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process or use different port
uvicorn app:app --port 8001
```

### Memory Issues

1. Use smaller batch sizes
2. Limit concurrent requests
3. Use model quantization
4. Increase container memory

## Performance Optimization

1. **Model Caching:** Models are loaded once at startup
2. **Batch Predictions:** Use `/predict/batch` for multiple predictions
3. **Async Operations:** API uses async/await for better concurrency
4. **Connection Pooling:** Use connection pooling for database (if added)

## Backup and Recovery

1. **Backup models:**
   ```bash
   tar -czf models_backup.tar.gz models/
   ```

2. **Backup configuration:**
   ```bash
   cp api/app.py api/app.py.backup
   ```

3. **Version control:**
   - Tag model versions
   - Keep model metadata
   - Document model changes

## Updates and Rollbacks

1. **Update API:**
   ```bash
   git pull
   docker-compose build
   docker-compose up -d
   ```

2. **Rollback:**
   ```bash
   git checkout <previous-version>
   docker-compose build
   docker-compose up -d
   ```

## Support

For issues or questions:
- Check logs: `docker-compose logs api`
- Review API docs: http://localhost:8000/docs
- Check health: http://localhost:8000/health

