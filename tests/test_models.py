"""
Unit tests for model training and prediction functions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.train_model import (
    prepare_data,
    train_random_forest,
    evaluate_model
)
from src.models.predict_model import predict, load_model


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Store': np.random.randint(1, 10, n_samples),
        'DayOfWeek': np.random.randint(1, 8, n_samples),
        'Sales': np.random.randint(1000, 10000, n_samples),
        'Customers': np.random.randint(100, 1000, n_samples),
        'Open': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'Promo': np.random.choice([0, 1], n_samples),
        'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], n_samples),
        'SchoolHoliday': np.random.choice([0, 1], n_samples),
        'Date': pd.date_range('2015-01-01', periods=n_samples, freq='D')
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model


# Tests for data preparation
class TestDataPreparation:
    
    def test_prepare_data_returns_correct_shapes(self, sample_data):
        """Test that prepare_data returns correct shapes."""
        X_train, X_test, y_train, y_test = prepare_data(
            sample_data,
            target_col='Sales',
            test_size=0.2
        )
        
        assert len(X_train) > len(X_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_prepare_data_excludes_target(self, sample_data):
        """Test that target column is not in features."""
        X_train, X_test, y_train, y_test = prepare_data(
            sample_data,
            target_col='Sales'
        )
        
        assert 'Sales' not in X_train.columns
        assert 'Sales' not in X_test.columns


# Tests for model training
class TestModelTraining:
    
    def test_train_random_forest_returns_model(self, sample_data):
        """Test that training returns a model object."""
        X_train, X_test, y_train, y_test = prepare_data(sample_data)
        
        params = {
            'n_estimators': 10,
            'max_depth': 5,
            'random_state': 42
        }
        
        model = train_random_forest(
            X_train, y_train,
            params=params,
            save_model=False
        )
        
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 10
    
    def test_model_can_predict(self, sample_data):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, y_test = prepare_data(sample_data)
        
        params = {'n_estimators': 10, 'random_state': 42}
        model = train_random_forest(
            X_train, y_train,
            params=params,
            save_model=False
        )
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(predictions >= 0)  # Sales should be non-negative


# Tests for model evaluation
class TestModelEvaluation:
    
    def test_evaluate_model_returns_metrics(self, sample_data):
        """Test that evaluation returns expected metrics."""
        X_train, X_test, y_train, y_test = prepare_data(sample_data)
        
        params = {'n_estimators': 10, 'random_state': 42}
        model = train_random_forest(
            X_train, y_train,
            params=params,
            save_model=False
        )
        
        metrics = evaluate_model(model, X_test, y_test)
        
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert 'RMSPE' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_metrics_are_reasonable(self, sample_data):
        """Test that metrics are in reasonable ranges."""
        X_train, X_test, y_train, y_test = prepare_data(sample_data)
        
        params = {'n_estimators': 10, 'random_state': 42}
        model = train_random_forest(
            X_train, y_train,
            params=params,
            save_model=False
        )
        
        metrics = evaluate_model(model, X_test, y_test)
        
        assert metrics['RMSE'] >= 0
        assert metrics['MAE'] >= 0
        assert -1 <= metrics['R2'] <= 1


# Tests for predictions
class TestPredictions:
    
    def test_predict_returns_correct_shape(self, sample_model):
        """Test that predict returns correct shape."""
        X = pd.DataFrame(np.random.rand(10, 5))
        
        predictions = predict(sample_model, X, return_dataframe=False)
        
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_returns_dataframe(self, sample_model):
        """Test that predict can return DataFrame."""
        X = pd.DataFrame(np.random.rand(10, 5))
        
        predictions = predict(sample_model, X, return_dataframe=True)
        
        assert isinstance(predictions, pd.DataFrame)
        assert 'Predicted_Sales' in predictions.columns
        assert len(predictions) == 10


# Tests for data validation
class TestDataValidation:
    
    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame is handled."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            prepare_data(empty_df)
    
    def test_missing_target_column_raises_error(self, sample_data):
        """Test that missing target column raises error."""
        df = sample_data.drop('Sales', axis=1)
        
        with pytest.raises(Exception):
            prepare_data(df, target_col='Sales')


# Integration tests
class TestIntegration:
    
    def test_full_training_pipeline(self, sample_data):
        """Test full training and prediction pipeline."""
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(sample_data)
        
        # Train model
        params = {'n_estimators': 10, 'random_state': 42}
        model = train_random_forest(
            X_train, y_train,
            params=params,
            save_model=False
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Predict
        predictions = predict(model, X_test, return_dataframe=False)
        
        assert len(predictions) == len(X_test)
        assert metrics['RMSE'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# TODO: Add tests for feature engineering
# TODO: Add tests for data preprocessing
# TODO: Add tests for LSTM models
# TODO: Add tests for API endpoints
# TODO: Add integration tests with real data samples

