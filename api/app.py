"""
FastAPI application for Rossmann Sales Forecasting API.

This API provides endpoints for:
- Health checks
- Making sales predictions
- Model information
- Batch predictions
- Store-specific multi-day predictions
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
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

# Import feature preparation utilities
from api.feature_preparation import (
    prepare_features_for_prediction,
    align_features_with_model,
    load_model_metadata
)

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
model_type = None  # 'xgboost', 'lstm', or 'other'
lstm_scaler_X = None
lstm_scaler_y = None
sequence_length = 30  # Default for LSTM


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
    Date: Optional[str] = Field(None, description="Date for prediction (YYYY-MM-DD)")
    CompetitionOpenSinceMonth: Optional[int] = Field(None, description="Month competition opened")
    CompetitionOpenSinceYear: Optional[int] = Field(None, description="Year competition opened")
    Promo2SinceWeek: Optional[int] = Field(None, description="Week Promo2 started")
    Promo2SinceYear: Optional[int] = Field(None, description="Year Promo2 started")
    PromoInterval: Optional[str] = Field(None, description="Promo2 interval")
    Customers: Optional[int] = Field(None, description="Number of customers (for feature engineering)")
    
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
                "Promo2": 0,
                "Date": "2015-09-18"
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
    metrics: Optional[Dict[str, Any]] = None


class StorePredictionRequest(BaseModel):
    """Request model for store-specific predictions."""
    store_id: int = Field(..., description="Store ID", ge=1)
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    n_weeks: int = Field(6, description="Number of weeks to predict", ge=1, le=12)


# Helper functions
def load_model_from_disk(model_path: Path = None):
    """Load model from disk (supports both .pkl and .h5 files)."""
    global loaded_model, model_metadata, model_type, lstm_scaler_X, lstm_scaler_y
    
    if model_path is None:
        # Try to find a model in the models directory
        # Exclude scaler files and other non-model files
        all_pkl_files = list(MODELS_DIR.glob("*.pkl"))
        all_h5_files = list(MODELS_DIR.glob("*.h5"))
        
        # Filter out scaler files and other non-model files
        exclude_patterns = ['scaler', 'metadata', 'comparison', 'evaluation']
        model_files = [
            f for f in all_pkl_files + all_h5_files
            if not any(pattern in f.name.lower() for pattern in exclude_patterns)
        ]
        
        # Prefer XGBoost models over LSTM for default loading
        xgb_models = [f for f in model_files if 'xgboost' in f.name.lower()]
        if xgb_models:
            model_path = xgb_models[0]
        elif model_files:
            model_path = model_files[0]
        else:
            logger.error("No model files found in models directory")
            return False
        
        logger.info(f"Auto-selected model: {model_path.name}")
    
    try:
        # Load metadata if available
        metadata = load_model_metadata(model_path)
        
        # Determine model type and load accordingly
        if model_path.suffix == '.h5':
            # LSTM model
            try:
                import tensorflow as tf
                from tensorflow import keras
                
                loaded_model = keras.models.load_model(model_path)
                model_type = 'lstm'
                
                # Try to load scalers
                scaler_X_path = MODELS_DIR / "lstm_scaler_X.pkl"
                scaler_y_path = MODELS_DIR / "lstm_scaler_y.pkl"
                
                if scaler_X_path.exists():
                    lstm_scaler_X = joblib.load(scaler_X_path)
                if scaler_y_path.exists():
                    lstm_scaler_y = joblib.load(scaler_y_path)
                
                logger.info(f"LSTM model loaded successfully: {model_path.name}")
            except ImportError:
                logger.error("TensorFlow not installed. Cannot load LSTM model.")
                return False
        else:
            # XGBoost or other sklearn-compatible model
            loaded_model = joblib.load(model_path)
            model_type = 'xgboost' if 'xgboost' in str(model_path).lower() else 'other'
            logger.info(f"Model loaded successfully: {model_path.name}")
        
        # Set metadata
        if metadata:
            model_metadata = metadata
        else:
            model_metadata = {
                "model_name": model_path.name,
                "model_type": model_type,
                "loaded_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        
        logger.info(f"Model type: {model_type}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def prepare_features(features: StoreFeatures) -> pd.DataFrame:
    """
    Prepare features for prediction using the feature engineering pipeline.
    
    Args:
        features: StoreFeatures object
    
    Returns:
        DataFrame with all engineered features aligned with model expectations
    """
    features_dict = features.dict(exclude_none=True)
    
    # Prepare features using the feature engineering pipeline
    df = prepare_features_for_prediction(
        features_dict,
        model_metadata=model_metadata if model_metadata else None,
        date=features_dict.get('Date')
    )
    
    # Align features with model expectations
    df = align_features_with_model(df, model_metadata if model_metadata else None)
    
    return df


def predict_with_model(features_df: pd.DataFrame) -> float:
    """
    Make prediction using the loaded model (supports both XGBoost and LSTM).
    
    Args:
        features_df: DataFrame with prepared features
    
    Returns:
        Predicted sales value
    """
    global loaded_model, model_type, lstm_scaler_X, lstm_scaler_y, sequence_length
    
    if model_type == 'lstm':
        # LSTM requires sequences
        try:
            import tensorflow as tf
            
            # For LSTM, we need to create a sequence
            # Since we only have one row, we'll pad/repeat it
            # This is a simplified approach - in production, you'd want historical data
            
            # Get feature values (exclude non-feature columns)
            exclude_cols = ['Date', 'Store', 'Sales', 'Customers']
            feature_cols = [col for col in features_df.columns 
                          if col not in exclude_cols]
            
            X = features_df[feature_cols].values
            
            # Scale features if scaler available
            if lstm_scaler_X is not None:
                X = lstm_scaler_X.transform(X)
            
            # Create sequence by repeating the single row
            # In production, you should use actual historical sequences
            X_seq = np.repeat(X.reshape(1, -1), sequence_length, axis=0)
            X_seq = X_seq.reshape(1, sequence_length, -1)
            
            # Make prediction
            y_pred_scaled = loaded_model.predict(X_seq, verbose=0)[0][0]
            
            # Inverse transform if scaler available
            if lstm_scaler_y is not None:
                y_pred = lstm_scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
            else:
                y_pred = y_pred_scaled
            
            return float(max(0, y_pred))
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            raise
    
    else:
        # XGBoost or other sklearn-compatible model
        try:
            # Final check: ensure all columns are numeric and Date is dropped
            if 'Date' in features_df.columns:
                features_df = features_df.drop(columns=['Date'])
            
            # Convert any remaining object columns to numeric
            for col in features_df.select_dtypes(include=['object']).columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
            
            # Ensure all columns are float
            features_df = features_df.astype(float)
            
            prediction = loaded_model.predict(features_df)[0]
            return float(max(0, prediction))
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            logger.error(f"Features DataFrame dtypes: {features_df.dtypes}")
            logger.error(f"Features DataFrame columns: {list(features_df.columns)}")
            raise


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
    
    try:
        features_count = None
        if model_metadata and 'feature_names' in model_metadata:
            features_count = len(model_metadata['feature_names'])
        elif model_metadata and 'data_info' in model_metadata and 'n_features' in model_metadata['data_info']:
            features_count = model_metadata['data_info']['n_features']
        
        metrics = None
        if model_metadata and 'metrics' in model_metadata:
            metrics = model_metadata['metrics']
            # Ensure metrics is a dict and handle NaN values
            if isinstance(metrics, dict):
                # Convert NaN to None for JSON serialization
                metrics = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) 
                           for k, v in metrics.items()}
        
        model_info = ModelInfo(
            model_name=model_metadata.get("model_name", "unknown") if model_metadata else "unknown",
            model_type=model_metadata.get("model_type", model_type or "unknown") if model_metadata else (model_type or "unknown"),
            version=model_metadata.get("version", "1.0.0") if model_metadata else "1.0.0",
            loaded_at=model_metadata.get("loaded_at") if model_metadata else None,
            features_count=features_count,
            metrics=metrics
        )
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return basic info even if there's an error
        return ModelInfo(
            model_name=model_metadata.get("model_name", "unknown") if model_metadata else "unknown",
            model_type=model_type or "unknown",
            version="1.0.0",
            loaded_at=None,
            features_count=None,
            metrics=None
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
        
        # Make prediction using appropriate model type
        prediction = predict_with_model(features_df)
        
        return PredictionResponse(
            store_id=request.features.Store,
            predicted_sales=prediction,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
            prediction = predict_with_model(features_df)
            
            predictions.append(
                PredictionResponse(
                    store_id=features.Store,
                    predicted_sales=prediction,
                    timestamp=datetime.now().isoformat()
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
        
        # Get model details
        models_info = []
        for model_file in model_files:
            model_path = MODELS_DIR / model_file
            metadata = load_model_metadata(model_path)
            
            model_info = {
                "name": model_file,
                "type": "LSTM" if model_file.endswith(".h5") else "XGBoost/ML",
                "has_metadata": metadata is not None
            }
            
            if metadata and 'metrics' in metadata:
                model_info["metrics"] = metadata['metrics']
            
            models_info.append(model_info)
        
        return {
            "available_models": model_files,
            "models_info": models_info,
            "count": len(model_files)
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.post("/predict/store/{store_id}", tags=["Prediction"])
async def predict_store_sales(
    store_id: int,
    request: StorePredictionRequest
):
    """
    Predict sales for a specific store over the next N weeks.
    
    Args:
        store_id: Store ID
        request: Request with start_date and n_weeks
    
    Returns:
        List of daily predictions
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Parse start date
        start_date = pd.to_datetime(request.start_date)
        
        # Generate date range
        dates = pd.date_range(start=start_date, periods=request.n_weeks * 7, freq='D')
        
        predictions = []
        
        # For each day, create features and predict
        # Note: This is a simplified version. In production, you'd want to:
        # 1. Use actual historical data for lag features
        # 2. Update lag features iteratively as you predict forward
        # 3. Handle store-specific characteristics better
        
        for date in dates:
            # Create basic features for this date
            features = StoreFeatures(
                Store=store_id,
                DayOfWeek=date.dayofweek + 1,  # Monday=1, Sunday=7
                Open=1,  # Assume store is open
                Promo=0,  # Default - should be provided or calculated
                StateHoliday="0",
                SchoolHoliday=0,  # Should be calculated based on date
                Date=date.strftime("%Y-%m-%d")
            )
            
            # Prepare features and predict
            features_df = prepare_features(features)
            prediction = predict_with_model(features_df)
            
            predictions.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_of_week": date.dayofweek + 1,
                "predicted_sales": prediction
            })
        
        return {
            "store_id": store_id,
            "start_date": request.start_date,
            "n_weeks": request.n_weeks,
            "predictions": predictions,
            "total_days": len(predictions),
            "average_daily_sales": float(np.mean([p["predicted_sales"] for p in predictions])),
            "total_weekly_sales": [
                float(sum([p["predicted_sales"] for p in predictions[i*7:(i+1)*7]]))
                for i in range(request.n_weeks)
            ]
        }
    
    except Exception as e:
        logger.error(f"Store prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Store prediction failed: {str(e)}")


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

