"""
FastAPI application for Rossmann Sales Forecasting API.

This API provides endpoints for:
- Health checks
- Making sales predictions
- Model information
- Batch predictions
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src import logger, MODELS_DIR

# Initialize FastAPI app
app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="API for predicting sales at Rossmann Pharmaceutical stores",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store loaded model
loaded_model = None
model_metadata = {}


# Pydantic models for request/response validation
class StoreFeatures(BaseModel):
    """Features for a single store prediction."""
    Store: int = Field(..., description="Store ID", ge=1)
    DayOfWeek: int = Field(..., description="Day of week (1-7)", ge=1, le=7)
    Open: int = Field(..., description="Store open (0 or 1)", ge=0, le=1)
    Promo: int = Field(..., description="Running promotion (0 or 1)", ge=0, le=1)
    StateHoliday: str = Field("0", description="State holiday (0, a, b, c)")
    SchoolHoliday: int = Field(0, description="School holiday (0 or 1)", ge=0, le=1)
    StoreType: Optional[str] = Field("a", description="Store type")
    Assortment: Optional[str] = Field("a", description="Assortment level")
    CompetitionDistance: Optional[float] = Field(None, description="Distance to competitor")
    Promo2: Optional[int] = Field(0, description="Continuing promotion", ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
                "Store": 1,
                "DayOfWeek": 5,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
                "StoreType": "c",
                "Assortment": "a",
                "CompetitionDistance": 1270.0,
                "Promo2": 0
            }
        }


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    features: StoreFeatures


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    predictions: List[StoreFeatures] = Field(..., description="List of feature sets")
    
    @validator('predictions')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    store_id: int
    predicted_sales: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_count: int


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    version: str
    loaded_at: Optional[str]
    features_count: Optional[int]


# Helper functions
def load_model_from_disk(model_path: Path = None):
    """Load model from disk."""
    global loaded_model, model_metadata
    
    if model_path is None:
        # Try to find a model in the models directory
        model_files = list(MODELS_DIR.glob("*.pkl"))
        if not model_files:
            logger.error("No model files found in models directory")
            return False
        model_path = model_files[0]  # Load the first model found
    
    try:
        loaded_model = joblib.load(model_path)
        model_metadata = {
            "model_name": model_path.name,
            "model_type": type(loaded_model).__name__,
            "loaded_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        logger.info(f"Model loaded successfully: {model_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def prepare_features(features: StoreFeatures) -> pd.DataFrame:
    """
    Prepare features for prediction.
    
    Note: This is a simplified version. In production, you should:
    1. Apply the same preprocessing pipeline used during training
    2. Handle feature engineering consistently
    3. Ensure all required features are present
    """
    features_dict = features.dict()
    df = pd.DataFrame([features_dict])
    
    # TODO: Apply proper feature engineering pipeline
    # TODO: Handle categorical encoding consistently with training
    # TODO: Add lag features if model was trained with them
    
    return df


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Rossmann Sales Forecasting API...")
    success = load_model_from_disk()
    if not success:
        logger.warning("No model loaded. Use /load-model endpoint to load a model.")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Rossmann Sales Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    model_loaded = loaded_model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if loaded_model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return ModelInfo(
        model_name=model_metadata.get("model_name", "unknown"),
        model_type=model_metadata.get("model_type", "unknown"),
        version=model_metadata.get("version", "unknown"),
        loaded_at=model_metadata.get("loaded_at"),
        features_count=None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sales(request: PredictionRequest):
    """
    Predict sales for a single store/day.
    
    Args:
        request: Prediction request with store features
    
    Returns:
        Predicted sales value
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features_df = prepare_features(request.features)
        
        # Make prediction
        prediction = loaded_model.predict(features_df)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        return PredictionResponse(
            store_id=request.features.Store,
            predicted_sales=float(prediction),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_sales_batch(request: BatchPredictionRequest):
    """
    Predict sales for multiple stores/days.
    
    Args:
        request: Batch prediction request
    
    Returns:
        List of predictions
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for features in request.predictions:
            features_df = prepare_features(features)
            prediction = loaded_model.predict(features_df)[0]
            prediction = max(0, prediction)
            
            predictions.append(
                PredictionResponse(
                    store_id=features.Store,
                    predicted_sales=float(prediction),
                    timestamp=datetime.now().isoformat()
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/model/load", tags=["Model"])
async def load_model_endpoint(model_name: str = Query(..., description="Model filename")):
    """
    Load a specific model from the models directory.
    
    Args:
        model_name: Name of the model file (e.g., 'xgboost.pkl')
    """
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_name}")
    
    success = load_model_from_disk(model_path)
    
    if success:
        return {
            "message": "Model loaded successfully",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")


@app.get("/models/list", tags=["Model"])
async def list_models():
    """List all available models in the models directory."""
    try:
        model_files = [f.name for f in MODELS_DIR.glob("*.pkl")]
        model_files += [f.name for f in MODELS_DIR.glob("*.h5")]
        
        return {
            "available_models": model_files,
            "count": len(model_files)
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


# TODO: Add authentication and API keys
# TODO: Add rate limiting
# TODO: Add request logging and monitoring
# TODO: Add model versioning
# TODO: Add prediction caching
# TODO: Add async prediction support for better performance

