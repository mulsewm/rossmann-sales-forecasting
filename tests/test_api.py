"""
Tests for the Rossmann Sales Forecasting API.
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_list_models():
    """Test list models endpoint."""
    response = client.get("/models/list")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "count" in data
    assert isinstance(data["available_models"], list)


def test_model_info_no_model():
    """Test model info endpoint when no model is loaded."""
    # This might fail if a model is auto-loaded on startup
    # In that case, we expect 200 with model info
    response = client.get("/model/info")
    # Either 404 (no model) or 200 (model loaded)
    assert response.status_code in [200, 404]


def test_predict_single():
    """Test single prediction endpoint."""
    # Create a valid prediction request
    request_data = {
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
            "Promo2": 0,
            "Date": "2015-09-18"
        }
    }
    
    response = client.post("/predict", json=request_data)
    
    # If model is not loaded, expect 503
    # If model is loaded, expect 200 with prediction
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "store_id" in data
        assert "predicted_sales" in data
        assert "timestamp" in data
        assert data["predicted_sales"] >= 0


def test_predict_batch():
    """Test batch prediction endpoint."""
    request_data = {
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
    
    response = client.post("/predict/batch", json=request_data)
    
    # If model is not loaded, expect 503
    # If model is loaded, expect 200 with predictions
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "total_count" in data
        assert data["total_count"] == 2
        assert len(data["predictions"]) == 2


def test_predict_batch_too_large():
    """Test batch prediction with too many items."""
    request_data = {
        "predictions": [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0
            }
        ] * 1001  # Exceeds limit of 1000
    }
    
    response = client.post("/predict/batch", json=request_data)
    assert response.status_code == 422  # Validation error


def test_predict_store():
    """Test store-specific prediction endpoint."""
    request_data = {
        "store_id": 1,
        "start_date": "2015-09-18",
        "n_weeks": 2
    }
    
    response = client.post("/predict/store/1", json=request_data)
    
    # If model is not loaded, expect 503
    # If model is loaded, expect 200 with predictions
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "store_id" in data
        assert "predictions" in data
        assert "total_days" in data
        assert len(data["predictions"]) == 14  # 2 weeks * 7 days


def test_invalid_store_features():
    """Test prediction with invalid features."""
    request_data = {
        "features": {
            "Store": 0,  # Invalid: must be >= 1
            "DayOfWeek": 5,
            "Open": 1,
            "Promo": 1
        }
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error


def test_load_model_endpoint():
    """Test load model endpoint."""
    # First, list available models
    list_response = client.get("/models/list")
    if list_response.status_code == 200:
        models = list_response.json()["available_models"]
        if models:
            # Try to load the first model
            model_name = models[0]
            response = client.post(f"/model/load?model_name={model_name}")
            # Either 200 (success) or 500 (error loading)
            assert response.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

