# User Guide - Rossmann Sales Forecasting

This guide provides step-by-step instructions for different user roles and use cases.

## Table of Contents

1. [For Data Scientists](#for-data-scientists)
2. [For ML Engineers](#for-ml-engineers)
3. [For Business Users](#for-business-users)
4. [For Developers](#for-developers)
5. [Troubleshooting](#troubleshooting)

---

## For Data Scientists

### Running the Complete Pipeline

#### 1. Data Exploration

```bash
# Start Jupyter Lab
jupyter lab

# Open and run: notebooks/01_data_exploration.ipynb
```

**What you'll learn:**
- Data distributions and patterns
- Missing value analysis
- Temporal trends
- Store characteristics

#### 2. Feature Engineering

```bash
# Open: notebooks/02_feature_engineering.ipynb
```

**Features Created:**
- 85 engineered features
- Lag features (1, 7, 14, 30 days)
- Rolling statistics (MA, std, min, max)
- Promotion and competition features
- Time-based and cyclical features

#### 3. Model Training

```bash
# Train all models
python scripts/train_pipeline.py --model all

# Train specific model
python scripts/train_pipeline.py --model xgboost

# Quick test mode
python scripts/train_pipeline.py --model xgboost --quick-mode
```

#### 4. Model Evaluation

```bash
# Open: notebooks/05_model_evaluation.ipynb
```

**Evaluation Metrics:**
- RMSPE (primary)
- RMSE, MAE, R²
- Feature importance
- Error analysis

### Experimenting with New Features

1. **Add feature in `src/features/feature_engineering.py`:**
   ```python
   def create_custom_feature(df: pd.DataFrame) -> pd.DataFrame:
       df['CustomFeature'] = df['Feature1'] * df['Feature2']
       return df
   ```

2. **Update feature pipeline:**
   ```python
   # In create_all_features()
   df = create_custom_feature(df)
   ```

3. **Retrain and evaluate:**
   ```bash
   python scripts/train_pipeline.py --model xgboost
   ```

### Hyperparameter Tuning

```python
# Example: Tune XGBoost
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 7, 8],
    'learning_rate': [0.05, 0.1, 0.15]
}

model = xgb.XGBRegressor()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
```

---

## For ML Engineers

### Deploying the API

#### Local Deployment

```bash
cd api
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Docker Deployment

```bash
# Build image
docker build -f api/Dockerfile -t rossmann-api .

# Run container
docker run -p 8000:8000 rossmann-api

# Or use Docker Compose
docker-compose -f api/docker-compose.yml up
```

#### Cloud Deployment

See [api/DEPLOYMENT.md](../api/DEPLOYMENT.md) for:
- AWS deployment
- GCP deployment
- Azure deployment

### Monitoring

#### Health Checks

```bash
curl http://localhost:8000/health
```

#### Model Information

```bash
curl http://localhost:8000/model/info
```

#### Logs

```bash
# Docker logs
docker-compose logs -f api

# Application logs
tail -f logs/rossmann_*.log
```

### Model Updates

#### Loading a New Model

```bash
# Via API
curl -X POST "http://localhost:8000/model/load?model_name=XGBoost_new.pkl"

# Or restart API with new model in models/ directory
```

#### Model Versioning

1. Save models with timestamps: `XGBoost_YYYYMMDD_HHMMSS.pkl`
2. Keep metadata files: `XGBoost_YYYYMMDD_HHMMSS_metadata.json`
3. Document changes in model metadata

---

## For Business Users

### Making Predictions via API

#### Single Prediction

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

prediction = response.json()
print(f"Predicted Sales: €{prediction['predicted_sales']:.2f}")
```

#### Store Forecast (6 Weeks)

```python
response = requests.post(
    "http://localhost:8000/predict/store/1",
    json={
        "start_date": "2015-09-18",
        "n_weeks": 6
    }
)

forecast = response.json()
print(f"Average Daily Sales: €{forecast['average_daily_sales']:.2f}")
print(f"Weekly Totals: {forecast['total_weekly_sales']}")
```

### Using the Web Interface

1. **Open API Documentation:**
   ```
   http://localhost:8000/docs
   ```

2. **Try it out:**
   - Click on `/predict` endpoint
   - Click "Try it out"
   - Enter your store features
   - Click "Execute"
   - View prediction result

### Understanding Predictions

#### Prediction Components

- **Base Sales:** Store's typical daily sales
- **Day of Week Effect:** Weekend vs weekday adjustments
- **Promotion Impact:** Additional sales from promotions
- **Holiday Effects:** Reductions during holidays
- **Seasonal Patterns:** Monthly and quarterly trends

#### Confidence in Predictions

- **High Confidence:** Normal business days, no promotions
- **Medium Confidence:** Promo days, weekends
- **Lower Confidence:** Holiday periods, unusual events

---

## For Developers

### Integrating the API

#### Python Client

```python
import requests
from typing import Dict, List

class RossmannAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, features: Dict) -> float:
        response = requests.post(
            f"{self.base_url}/predict",
            json={"features": features}
        )
        response.raise_for_status()
        return response.json()["predicted_sales"]
    
    def predict_batch(self, features_list: List[Dict]) -> List[float]:
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"predictions": features_list}
        )
        response.raise_for_status()
        return [p["predicted_sales"] for p in response.json()["predictions"]]

# Usage
client = RossmannAPIClient()
sales = client.predict({
    "Store": 1,
    "DayOfWeek": 5,
    "Open": 1,
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 0
})
```

#### JavaScript/Node.js Client

```javascript
const axios = require('axios');

async function predictSales(features) {
    const response = await axios.post('http://localhost:8000/predict', {
        features: features
    });
    return response.data.predicted_sales;
}

// Usage
const sales = await predictSales({
    Store: 1,
    DayOfWeek: 5,
    Open: 1,
    Promo: 1,
    StateHoliday: "0",
    SchoolHoliday: 0
});
```

### Error Handling

```python
import requests
from requests.exceptions import RequestException

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": {...}},
        timeout=5
    )
    response.raise_for_status()
    prediction = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
    print(f"Response: {e.response.text}")
except RequestException as e:
    print(f"Request failed: {e}")
```

### Testing

```bash
# Run API tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=api --cov-report=html
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Problem:** API shows "Model not loaded"

**Solutions:**
```bash
# Check if model files exist
ls models/*.pkl models/*.h5

# Check model path in logs
grep "Model loaded" logs/rossmann_*.log

# Manually load model
curl -X POST "http://localhost:8000/model/load?model_name=XGBoost_20251126_161803.pkl"
```

#### 2. Feature Mismatch Error

**Problem:** "Feature mismatch" or "Missing features"

**Solutions:**
- Check model metadata: `GET /model/info`
- Ensure all required features are provided
- Verify feature names match training data

#### 3. API Not Responding

**Problem:** Connection refused or timeout

**Solutions:**
```bash
# Check if API is running
curl http://localhost:8000/health

# Check port availability
lsof -i :8000

# Restart API
uvicorn app:app --reload
```

#### 4. Low Prediction Accuracy

**Problem:** Predictions seem inaccurate

**Solutions:**
- Verify input features are correct
- Check if model was trained on similar data
- Ensure date features match prediction date
- Review model performance metrics

#### 5. Docker Issues

**Problem:** Container won't start

**Solutions:**
```bash
# Check logs
docker-compose logs api

# Rebuild image
docker-compose build --no-cache

# Check volumes are mounted
docker-compose config
```

### Getting Help

1. **Check Documentation:**
   - [README.md](../README.md)
   - [API README](../api/README.md)
   - [Deployment Guide](../api/DEPLOYMENT.md)

2. **Review Logs:**
   ```bash
   tail -f logs/rossmann_*.log
   ```

3. **Test API:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/model/info
   ```

4. **Check Model Files:**
   ```bash
   ls -lh models/
   ```

---

## Quick Reference

### API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/models/list` | GET | List available models |
| `/model/load` | POST | Load specific model |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/predict/store/{id}` | POST | Store forecast |

### Key Files

| File | Purpose |
|------|---------|
| `notebooks/01_data_exploration.ipynb` | EDA |
| `notebooks/02_feature_engineering.ipynb` | Feature creation |
| `notebooks/03_ml_modeling.ipynb` | ML models |
| `notebooks/04_deep_learning_lstm.ipynb` | LSTM model |
| `notebooks/05_model_evaluation.ipynb` | Model evaluation |
| `scripts/train_pipeline.py` | Training script |
| `api/app.py` | API application |
| `api/feature_preparation.py` | Feature engineering |

### Common Commands

```bash
# Start API
uvicorn app:app --reload

# Train model
python scripts/train_pipeline.py --model xgboost

# Run tests
pytest tests/ -v

# Check health
curl http://localhost:8000/health
```

---

**Last Updated:** November 2025  
**Version:** 1.0

