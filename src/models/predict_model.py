"""
Prediction module for making forecasts with trained models.
"""

import sys
from pathlib import Path
from typing import Union, Optional, Any
import pandas as pd
import numpy as np
import joblib
import argparse
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import logger, MODELS_DIR


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded model
    
    Raises:
        FileNotFoundError: If model file not found
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def predict(
    model: Any,
    X: pd.DataFrame,
    return_dataframe: bool = True
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        X: Features DataFrame
        return_dataframe: Whether to return predictions as DataFrame
    
    Returns:
        Predictions as numpy array or DataFrame
    """
    logger.info(f"Making predictions for {len(X)} samples...")
    
    try:
        predictions = model.predict(X)
        logger.info("Predictions completed successfully")
        
        if return_dataframe:
            pred_df = pd.DataFrame({
                'Predicted_Sales': predictions
            }, index=X.index)
            return pred_df
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def predict_from_file(
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load model and data, make predictions, and optionally save results.
    
    Args:
        model_path: Path to saved model
        data_path: Path to input data CSV
        output_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    logger.info("Starting prediction pipeline...")
    
    # Load model
    model = load_model(model_path)
    
    # Load data
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Prepare features (exclude non-feature columns)
    exclude_cols = ['Date', 'Store', 'Sales', 'Customers']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle missing feature columns (model might expect more features)
    # This is a simplified version - in production, use the same preprocessing pipeline
    X = df[feature_cols] if feature_cols else df.select_dtypes(include=[np.number])
    
    # Make predictions
    predictions = predict(model, X, return_dataframe=False)
    
    # Add predictions to dataframe
    df['Predicted_Sales'] = predictions
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")
    
    return df


def predict_sales_for_store(
    model: Any,
    store_id: int,
    dates: pd.DatetimeIndex,
    features: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict sales for a specific store over a date range.
    
    Args:
        model: Trained model
        store_id: Store ID
        dates: Date range for predictions
        features: Pre-computed features for the store
    
    Returns:
        DataFrame with date and predicted sales
    """
    logger.info(f"Predicting sales for Store {store_id} over {len(dates)} days")
    
    # Filter features for the specific store and dates
    store_features = features[
        (features['Store'] == store_id) & 
        (features['Date'].isin(dates))
    ].copy()
    
    if len(store_features) == 0:
        logger.warning(f"No features found for Store {store_id}")
        return pd.DataFrame()
    
    # Prepare features for prediction
    exclude_cols = ['Date', 'Store', 'Sales', 'Customers']
    X = store_features[[col for col in store_features.columns 
                       if col not in exclude_cols]]
    
    # Make predictions
    predictions = predict(model, X, return_dataframe=False)
    
    # Create result dataframe
    result = pd.DataFrame({
        'Store': store_id,
        'Date': store_features['Date'].values,
        'Predicted_Sales': predictions
    })
    
    return result


def predict_next_n_weeks(
    model: Any,
    store_id: int,
    start_date: pd.Timestamp,
    n_weeks: int = 6,
    historical_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Predict sales for the next N weeks for a specific store.
    
    Note: This is a simplified version. In practice, you'll need to:
    1. Generate features for future dates
    2. Use lag features from historical data
    3. Update lag features iteratively as you predict
    
    Args:
        model: Trained model
        store_id: Store ID
        start_date: Starting date for predictions
        n_weeks: Number of weeks to predict
        historical_data: Historical data for lag feature calculation
    
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Predicting next {n_weeks} weeks for Store {store_id}")
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_weeks*7, freq='D')
    
    # TODO: Implement proper feature generation for future dates
    # TODO: Handle lag features using iterative prediction
    # TODO: Incorporate store-specific characteristics
    
    logger.warning("This is a placeholder implementation")
    logger.warning("Implement proper feature engineering for production use")
    
    result = pd.DataFrame({
        'Store': store_id,
        'Date': dates,
        'Predicted_Sales': np.zeros(len(dates))  # Placeholder
    })
    
    return result


def main():
    """
    Command-line interface for making predictions.
    
    Example:
        python -m src.models.predict_model \
            --model models/xgboost.pkl \
            --data data/test.csv \
            --output predictions.csv
    """
    parser = argparse.ArgumentParser(description='Make sales predictions')
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to input data CSV'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save predictions (optional)'
    )
    
    args = parser.parse_args()
    
    # Make predictions
    predictions_df = predict_from_file(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path
    )
    
    logger.info("\nPrediction Summary:")
    logger.info(f"Total predictions: {len(predictions_df)}")
    logger.info(f"Mean predicted sales: {predictions_df['Predicted_Sales'].mean():.2f}")
    logger.info(f"Min predicted sales: {predictions_df['Predicted_Sales'].min():.2f}")
    logger.info(f"Max predicted sales: {predictions_df['Predicted_Sales'].max():.2f}")


if __name__ == "__main__":
    main()


# TODO: Add confidence intervals for predictions
# TODO: Add prediction explanations using SHAP
# TODO: Add batch prediction support
# TODO: Add API endpoint integration

