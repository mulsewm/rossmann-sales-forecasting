"""
Complete training pipeline for Rossmann sales forecasting.

This script performs:
1. Data loading and preprocessing
2. Feature engineering
3. Model training (Random Forest, XGBoost, LightGBM)
4. Model evaluation
5. Model saving

Usage:
    python scripts/train_pipeline.py --model all
    python scripts/train_pipeline.py --model xgboost --save-predictions
"""

import sys
from pathlib import Path
import argparse
import time
from typing import List
import pandas as pd
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src import logger, MODELS_DIR, DATA_DIR
from src.data.load_data import load_all_data
from src.data.preprocessing import (
    clean_data,
    handle_missing_values,
    merge_store_data,
    create_time_features
)
from src.features.feature_engineering import create_all_features
from src.models.train_model import (
    prepare_data,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    evaluate_model,
    get_feature_importance
)


def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Rossmann sales forecasting models'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['rf', 'xgboost', 'lightgbm', 'all'],
        default='all',
        help='Model to train (default: all)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    
    parser.add_argument(
        '--quick-mode',
        action='store_true',
        help='Quick training with reduced parameters for testing'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to CSV'
    )
    
    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help='Generate feature importance plots'
    )
    
    return parser


def load_and_preprocess_data():
    """Load and preprocess all data."""
    logger.info("="*60)
    logger.info("STEP 1: Loading Data")
    logger.info("="*60)
    
    # Load data
    train_df, test_df, store_df = load_all_data()
    
    # Merge with store data
    logger.info("Merging with store data...")
    train_df = merge_store_data(train_df, store_df)
    
    # Clean data
    logger.info("Cleaning data...")
    train_df = clean_data(train_df, data_type='train')
    
    # Handle missing values
    logger.info("Handling missing values...")
    train_df = handle_missing_values(train_df)
    
    logger.info(f"✓ Data loaded: {train_df.shape}")
    
    return train_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("="*60)
    
    # Create time features
    logger.info("Creating time-based features...")
    df = create_time_features(df)
    
    # Create all features
    logger.info("Creating engineered features...")
    df = create_all_features(
        df,
        include_lag=True,
        include_rolling=True,
        include_promo=True,
        include_competition=True
    )
    
    # Drop rows with NaN values created by lag/rolling features
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    logger.info(f"Dropped {dropped_rows:,} rows with missing values from lag features")
    logger.info(f"✓ Feature engineering completed: {df.shape}")
    
    return df


def train_models(
    X_train, X_test, y_train, y_test,
    model_list: List[str],
    quick_mode: bool = False
):
    """Train specified models."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Model Training")
    logger.info("="*60)
    
    results = {}
    
    # Random Forest
    if 'rf' in model_list or 'all' in model_list:
        logger.info("\nTraining Random Forest...")
        start_time = time.time()
        
        params = {
            'n_estimators': 50 if quick_mode else 200,
            'max_depth': 15 if quick_mode else 25,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }
        
        rf_model = train_random_forest(
            X_train, y_train,
            params=params,
            save_model=True,
            model_name='random_forest.pkl'
        )
        
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        results['Random Forest'] = {
            'model': rf_model,
            'metrics': rf_metrics,
            'training_time': time.time() - start_time
        }
    
    # XGBoost
    if 'xgboost' in model_list or 'all' in model_list:
        logger.info("\nTraining XGBoost...")
        start_time = time.time()
        
        # Use validation set for early stopping
        X_train_split = X_train[:int(0.8*len(X_train))]
        X_val_split = X_train[int(0.8*len(X_train)):]
        y_train_split = y_train[:int(0.8*len(y_train))]
        y_val_split = y_train[int(0.8*len(y_train)):]
        
        params = {
            'n_estimators': 500 if quick_mode else 2000,
            'max_depth': 8 if quick_mode else 12,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
        
        xgb_model = train_xgboost(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            params=params,
            save_model=True,
            model_name='xgboost.pkl'
        )
        
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        results['XGBoost'] = {
            'model': xgb_model,
            'metrics': xgb_metrics,
            'training_time': time.time() - start_time
        }
    
    # LightGBM
    if 'lightgbm' in model_list or 'all' in model_list:
        logger.info("\nTraining LightGBM...")
        start_time = time.time()
        
        X_train_split = X_train[:int(0.8*len(X_train))]
        X_val_split = X_train[int(0.8*len(X_train)):]
        y_train_split = y_train[:int(0.8*len(y_train))]
        y_val_split = y_train[int(0.8*len(y_train)):]
        
        params = {
            'n_estimators': 500 if quick_mode else 2000,
            'max_depth': 8 if quick_mode else 12,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
        
        lgb_model = train_lightgbm(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            params=params,
            save_model=True,
            model_name='lightgbm.pkl'
        )
        
        lgb_metrics = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        results['LightGBM'] = {
            'model': lgb_model,
            'metrics': lgb_metrics,
            'training_time': time.time() - start_time
        }
    
    return results


def compare_results(results: dict):
    """Compare and display results from all models."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Model Comparison")
    logger.info("="*60)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, result in results.items():
        metrics = result['metrics']
        metrics['Model'] = model_name
        metrics['Training_Time'] = result['training_time']
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df[['Model', 'RMSE', 'MAE', 'R2', 'RMSPE', 'Training_Time']]
    
    logger.info("\nModel Performance Comparison:")
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['RMSPE'].idxmin(), 'Model']
    logger.info(f"\n🏆 Best Model (by RMSPE): {best_model_name}")
    
    # Save comparison
    comparison_path = MODELS_DIR / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"✓ Comparison saved to: {comparison_path}")
    
    return best_model_name


def main():
    """Main training pipeline."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("ROSSMANN SALES FORECASTING - TRAINING PIPELINE")
    logger.info("="*60)
    logger.info(f"Model(s): {args.model}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Quick mode: {args.quick_mode}")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load and preprocess data
        train_df = load_and_preprocess_data()
        
        # Step 2: Feature engineering
        train_df = engineer_features(train_df)
        
        # Step 3: Prepare for training
        X_train, X_test, y_train, y_test = prepare_data(
            train_df,
            target_col='Sales',
            test_size=args.test_size
        )
        
        # Step 4: Train models
        model_list = [args.model] if args.model != 'all' else ['rf', 'xgboost', 'lightgbm']
        results = train_models(
            X_train, X_test, y_train, y_test,
            model_list=model_list,
            quick_mode=args.quick_mode
        )
        
        # Step 5: Compare results
        best_model_name = compare_results(results)
        
        # Optional: Save predictions
        if args.save_predictions:
            logger.info("\nSaving predictions...")
            best_model = results[best_model_name]['model']
            predictions = best_model.predict(X_test)
            
            pred_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': predictions,
                'Error': y_test - predictions
            })
            
            pred_path = MODELS_DIR / 'predictions.csv'
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"✓ Predictions saved to: {pred_path}")
        
        # Optional: Feature importance
        if args.feature_importance:
            logger.info("\nGenerating feature importance...")
            for model_name, result in results.items():
                importance_df = get_feature_importance(
                    result['model'],
                    X_train.columns.tolist(),
                    top_n=20
                )
                
                importance_path = MODELS_DIR / f'{model_name.lower().replace(" ", "_")}_feature_importance.csv'
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"✓ {model_name} importance saved to: {importance_path}")
        
        # Summary
        total_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Models saved in: {MODELS_DIR}")
        logger.info("\nNext steps:")
        logger.info("1. Review model performance in notebooks/05_model_evaluation.ipynb")
        logger.info("2. Start API server: cd api && uvicorn app:app --reload")
        logger.info("3. Make predictions: python -m src.models.predict_model")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

