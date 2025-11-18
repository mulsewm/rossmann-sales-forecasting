"""
Model training functions for Rossmann sales forecasting.

Supports various ML models:
- Random Forest
- XGBoost
- LightGBM
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import logger, MODELS_DIR


def prepare_data(
    df: pd.DataFrame,
    target_col: str = 'Sales',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for training.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Fraction of data for testing
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Preparing data for training...")
    
    # Remove non-numeric columns
    exclude_cols = ['Date', 'Store']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle categorical columns
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
    model_name: str = 'random_forest.pkl'
) -> RandomForestRegressor:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Model hyperparameters
        save_model: Whether to save the trained model
        model_name: Name of the saved model file
    
    Returns:
        Trained Random Forest model
    """
    logger.info("Training Random Forest model...")
    
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }
    
    logger.info(f"Parameters: {params}")
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    logger.info("Training completed")
    
    if save_model:
        model_path = MODELS_DIR / model_name
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
    
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
    model_name: str = 'xgboost.pkl'
) -> xgb.XGBRegressor:
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: Model hyperparameters
        save_model: Whether to save the trained model
        model_name: Name of the saved model file
    
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model...")
    
    if params is None:
        params = {
            'n_estimators': 1000,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
    
    logger.info(f"Parameters: {params}")
    
    model = xgb.XGBRegressor(**params)
    
    # Setup early stopping if validation set provided
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=50,
        verbose=True
    )
    
    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info("Training completed")
    
    if save_model:
        model_path = MODELS_DIR / model_name
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
    
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
    model_name: str = 'lightgbm.pkl'
) -> lgb.LGBMRegressor:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: Model hyperparameters
        save_model: Whether to save the trained model
        model_name: Name of the saved model file
    
    Returns:
        Trained LightGBM model
    """
    logger.info("Training LightGBM model...")
    
    if params is None:
        params = {
            'n_estimators': 1000,
            'max_depth': 10,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
    
    logger.info(f"Parameters: {params}")
    
    model = lgb.LGBMRegressor(**params)
    
    # Setup callbacks
    callbacks = []
    if X_val is not None and y_val is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
        callbacks.append(lgb.log_evaluation(period=10))
        eval_set = [(X_val, y_val)]
    else:
        eval_set = None
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=callbacks
    )
    
    logger.info("Training completed")
    
    if save_model:
        model_path = MODELS_DIR / model_name
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
    
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name for logging
    
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # RMSPE (Root Mean Square Percentage Error) - Kaggle metric
    rmspe = np.sqrt(np.mean(((y_test - y_pred) / y_test) ** 2)) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'RMSPE': rmspe
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"{model_name} Performance Metrics:")
    logger.info(f"{'='*50}")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info(f"{'='*50}\n")
    
    return metrics


def get_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance
    """
    logger.info("Extracting feature importance...")
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    logger.info(f"\nTop {top_n} Features:")
    for idx, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.6f}")
    
    return importance_df


# TODO: Add hyperparameter tuning functions
# TODO: Add cross-validation
# TODO: Add ensemble methods
# TODO: Add model comparison utilities

