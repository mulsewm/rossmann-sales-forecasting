# Rossmann Sales Forecasting API

Production-ready FastAPI application for serving sales predictions.

## Features

-  Single and batch predictions
-  Support for XGBoost and LSTM models
-  Store-specific multi-week forecasts
-  Comprehensive feature engineering pipeline
-  Model metadata and information endpoints
-  Health checks and monitoring
-  Docker support
-  Comprehensive test suite

## Quick Start

### Local Development

```bash
# Install dependencies
cd api
pip install -r requirements.txt
pip install -r ../requirements.txt

# Start the API
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

### Docker

```bash
# Build and run
docker-compose -f api/docker-compose.yml up --build

# Or using Docker directly
docker build -f api/Dockerfile -t rossmann-api .
docker run -p 8000:8000 rossmann-api
```

## API Endpoints

### Health & Info

- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model information
- `GET /models/list` - List available models
- `POST /model/load?model_name=<name>` - Load a specific model

### Predictions

- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions (up to 1000)
- `POST /predict/store/{store_id}` - Multi-week forecast for a store

## Example Usage

### Single Prediction

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": {
            "Store": 1,
            "DayOfWeek": 5,
            "Open": 1,
            "Promo": 1,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
            "StoreType": "c",
            "Assortment": "a",
            "CompetitionDistance": 1270.0,
            "Date": "2015-09-18"
        }
    }
)

print(response.json())
# {
#   "store_id": 1,
#   "predicted_sales": 5263.84,
#   "timestamp": "2025-11-27T10:30:00"
# }
```

### Batch Prediction

```python
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "predictions": [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0
            },
            {
                "Store": 2,
                "DayOfWeek": 6,
                "Open": 1,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0
            }
        ]
    }
)
```

### Store Forecast

```python
response = requests.post(
    "http://localhost:8000/predict/store/1",
    json={
        "start_date": "2015-09-18",
        "n_weeks": 6
    }
)
```

## Model Support

### XGBoost Models (.pkl)

- Automatically loads from `models/` directory
- Supports model metadata for feature alignment
- Fast inference (< 50ms per prediction)

### LSTM Models (.h5)

- Requires TensorFlow
- Uses sequence-based predictions
- Supports feature and target scalers

## Feature Engineering

The API automatically applies the same feature engineering pipeline used during training:

- Time-based features (year, month, day, week, etc.)
- Cyclical encoding (sin/cos for day of week, month)
- Promotion features
- Competition features
- Holiday features
- One-hot encoding for categorical variables

**Note:** Lag and rolling features are set to default values for single predictions since historical data is not available. For more accurate predictions, use the batch endpoint with historical context.

## Testing

```bash
# Run API tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=api --cov-report=html
```

## Configuration

Set environment variables or create a `.env` file:

```bash
API_HOST=0.0.0.0
API_PORT=8000
MODELS_DIR=models
LOG_LEVEL=INFO
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions including:

- Local deployment
- Docker deployment
- Cloud platform deployment (AWS, GCP, Azure)
- Load balancing
- Monitoring
- Security best practices

## Architecture

```
api/
├── app.py                 # Main FastAPI application
├── feature_preparation.py # Feature engineering utilities
├── requirements.txt       # API dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose config
├── DEPLOYMENT.md         # Deployment guide
└── README.md            # This file
```

## Performance

- **Response Time:** < 100ms for single predictions
- **Throughput:** ~100 requests/second (depends on hardware)
- **Batch Processing:** Optimized for up to 1000 predictions per request

## Security

For production deployment:

1. Add API key authentication
2. Enable HTTPS/TLS
3. Implement rate limiting
4. Add request validation
5. Set up monitoring and alerting

See [DEPLOYMENT.md](DEPLOYMENT.md) for security best practices.

## Troubleshooting

### Model Not Loading

1. Check model files exist: `ls models/`
2. Check logs: `docker-compose logs api`
3. Verify model format: `python -c "import joblib; joblib.load('models/XGBoost_*.pkl')"`

### Feature Mismatch

If you get feature mismatch errors:

1. Check model metadata: `GET /model/info`
2. Verify feature names match training
3. Ensure all required features are provided

### LSTM Model Issues

1. Ensure TensorFlow is installed: `pip install tensorflow`
2. Check scaler files exist: `ls models/lstm_scaler_*.pkl`
3. Verify sequence length matches training

## Support

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Model Info: http://localhost:8000/model/info

## License

MIT License - see LICENSE file for details.

