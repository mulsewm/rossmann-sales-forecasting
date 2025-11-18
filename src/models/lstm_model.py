"""
LSTM Deep Learning model for Rossmann sales forecasting.

This module implements LSTM-based models for time series forecasting.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM models will not work.")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import logger, MODELS_DIR


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Feature array
        target: Target array
        sequence_length: Number of time steps to look back
    
    Returns:
        X (sequences), y (targets)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(target[i + sequence_length])
    
    return np.array(X), np.array(y)


def prepare_lstm_data(
    df: pd.DataFrame,
    target_col: str = 'Sales',
    sequence_length: int = 30,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    """
    Prepare data for LSTM model.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        sequence_length: Number of time steps
        test_size: Fraction for test set
    
    Returns:
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler
    """
    from sklearn.preprocessing import StandardScaler
    
    logger.info("Preparing data for LSTM...")
    
    # Separate features and target
    exclude_cols = ['Date', 'Store', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Scale features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    
    # Split into train and test
    split_idx = int(len(X_seq) * (1 - test_size))
    
    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]
    
    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Test sequences: {X_test.shape}")
    logger.info(f"Features per timestep: {X_train.shape[2]}")
    
    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler


def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: list = [128, 64],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Build LSTM model architecture.
    
    Args:
        input_shape: Shape of input (sequence_length, n_features)
        lstm_units: List of LSTM layer units
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed. Install with: pip install tensorflow")
    
    logger.info("Building LSTM model...")
    
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        lstm_units[0],
        input_shape=input_shape,
        return_sequences=len(lstm_units) > 1
    ))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    # Additional LSTM layers
    for i, units in enumerate(lstm_units[1:]):
        return_seq = i < len(lstm_units) - 2
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
    
    # Dense output layer
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Model built with {model.count_params():,} parameters")
    model.summary(print_fn=logger.info)
    
    return model


def train_lstm_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 64,
    model_name: str = 'lstm_model.h5'
) -> keras.callbacks.History:
    """
    Train LSTM model.
    
    Args:
        model: Keras model
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences (optional)
        y_val: Validation targets (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        model_name: Name for saved model
    
    Returns:
        Training history
    """
    logger.info("Training LSTM model...")
    
    # Setup callbacks
    model_path = MODELS_DIR / model_name
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(model_path),
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info(f"Training completed. Model saved to: {model_path}")
    
    return history


def evaluate_lstm_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: Any
) -> Dict[str, float]:
    """
    Evaluate LSTM model performance.
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: Test targets
        target_scaler: Scaler used for target variable
    
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating LSTM model...")
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # RMSPE
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'RMSPE': rmspe
    }
    
    logger.info(f"\n{'='*50}")
    logger.info("LSTM Model Performance Metrics:")
    logger.info(f"{'='*50}")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info(f"{'='*50}\n")
    
    return metrics


def load_lstm_model(model_path: Path) -> keras.Model:
    """
    Load a saved LSTM model.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed")
    
    logger.info(f"Loading LSTM model from: {model_path}")
    model = keras_load_model(model_path)
    logger.info("Model loaded successfully")
    
    return model


def predict_lstm(
    model: keras.Model,
    X: np.ndarray,
    target_scaler: Any
) -> np.ndarray:
    """
    Make predictions with LSTM model.
    
    Args:
        model: Trained LSTM model
        X: Input sequences
        target_scaler: Scaler for inverse transformation
    
    Returns:
        Predictions
    """
    logger.info(f"Making predictions for {len(X)} sequences...")
    
    y_pred_scaled = model.predict(X)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    
    return y_pred


# TODO: Add attention mechanism to LSTM
# TODO: Add bidirectional LSTM option
# TODO: Add GRU as an alternative
# TODO: Add multi-step ahead forecasting
# TODO: Add prediction intervals

