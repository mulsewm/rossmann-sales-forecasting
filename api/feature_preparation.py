"""
Feature preparation utilities for API predictions.

This module handles feature engineering for single predictions,
ensuring consistency with the training pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import create_time_features, handle_missing_values
from src.features.feature_engineering import (
    create_promotion_features,
    create_competition_features,
    create_holiday_features,
    create_customer_features
)


def load_model_metadata(model_path: Path) -> Optional[Dict]:
    """Load model metadata if available."""
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                content = f.read()
                # Replace NaN (invalid JSON) with null before parsing
                content = content.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
                metadata = json.loads(content)
                
                # Handle NaN values (convert to None)
                def clean_nan(obj):
                    if isinstance(obj, dict):
                        return {k: clean_nan(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_nan(item) for item in obj]
                    elif isinstance(obj, float) and np.isnan(obj):
                        return None
                    elif isinstance(obj, str) and obj.lower() == 'nan':
                        return None
                    return obj
                return clean_nan(metadata)
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing metadata file {metadata_path}: {e}")
            return None
    return None


def prepare_features_for_prediction(
    features_dict: Dict,
    model_metadata: Optional[Dict] = None,
    date: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare features for a single prediction.
    
    This function creates all necessary features matching the training pipeline,
    but without lag features (which require historical data).
    
    Args:
        features_dict: Dictionary of input features
        model_metadata: Model metadata containing expected feature names
        date: Date string for the prediction (defaults to today)
    
    Returns:
        DataFrame with all engineered features
    """
    # Create DataFrame from input features
    df = pd.DataFrame([features_dict])
    
    # Add Date if not provided
    if 'Date' not in df.columns:
        if date:
            df['Date'] = pd.to_datetime(date)
        else:
            df['Date'] = pd.Timestamp.now()
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Fill missing values with defaults
    defaults = {
        'CompetitionDistance': df['CompetitionDistance'].median() if 'CompetitionDistance' in df.columns else 0,
        'CompetitionOpenSinceMonth': 0,
        'CompetitionOpenSinceYear': 0,
        'Promo2SinceWeek': 0,
        'Promo2SinceYear': 0,
        'PromoInterval': 'None',
        'Customers': 0,  # Will be needed for some features
    }
    
    for key, value in defaults.items():
        if key not in df.columns:
            df[key] = value
        elif df[key].isna().any():
            df[key].fillna(value, inplace=True)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create time-based features
    df = create_time_features(df)
    
    # Create promotion features
    df = create_promotion_features(df)
    
    # Create competition features
    df = create_competition_features(df)
    
    # Create holiday features
    df = create_holiday_features(df)
    
    # Create customer features (if Customers available)
    if 'Customers' in df.columns:
        df = create_customer_features(df)
    
    # Create one-hot encoded features for categorical variables
    categorical_cols = ['StoreType', 'Assortment', 'StateHoliday', 'CompDistanceCategory']
    
    for col in categorical_cols:
        if col in df.columns:
            # Create one-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            # Keep original column for now (will be removed later if needed)
    
    # Create additional derived features
    # WeekOfYear (already created by create_time_features, but ensure it's there)
    if 'Week' in df.columns:
        df['WeekOfYear'] = df['Week']
    
    # Cyclical encoding for DayOfWeek and Month
    if 'DayOfWeek' in df.columns:
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    if 'Month' in df.columns:
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Additional features that don't require historical data
    if 'Month' in df.columns:
        df['IsMonthMiddle'] = ((df['Day'] >= 10) & (df['Day'] <= 20)).astype(int)
    
    if 'Date' in df.columns:
        df['DaysInMonth'] = df['Date'].dt.days_in_month
    
    # Promo features
    if 'Promo' in df.columns:
        df['PromoActive'] = df['Promo'].astype(int)
    
    # Competition features
    if 'CompetitionDistance' in df.columns:
        df['HasCompetition'] = (df['CompetitionDistance'].notna() & (df['CompetitionDistance'] > 0)).astype(int)
    
    # For lag and rolling features, we'll set them to 0 or mean values
    # since we don't have historical data for single predictions
    lag_features = ['Sales_Lag1', 'Sales_Lag7', 'Sales_Lag14', 'Sales_Lag30',
                   'Customers_Lag1', 'Customers_Lag7']
    
    rolling_features = ['Sales_MA7', 'Sales_MA14', 'Sales_MA30',
                      'Sales_Std7', 'Sales_Std30', 'Sales_Min7', 'Sales_Max7',
                      'Customers_MA7', 'Customers_MA30',
                      'SalesPerCustomer_MA7',
                      'Sales_Trend7', 'Sales_Trend30']
    
    for feat in lag_features + rolling_features:
        if feat not in df.columns:
            df[feat] = 0.0
    
    # If model metadata is available, ensure all expected features are present
    if model_metadata and 'feature_names' in model_metadata:
        expected_features = model_metadata['feature_names']
        
        # Add missing features with default values
        for feat in expected_features:
            if feat not in df.columns:
                # Set default based on feature type
                if 'Lag' in feat or 'MA' in feat or 'Trend' in feat or 'Std' in feat:
                    df[feat] = 0.0
                elif 'Min' in feat:
                    df[feat] = 0.0
                elif 'Max' in feat:
                    df[feat] = 10000.0  # Reasonable default for sales max
                else:
                    df[feat] = 0
        
        # Select only expected features in the correct order
        df = df[expected_features]
    
    # Convert categorical object columns to numeric (label encoding)
    # MonthName: Convert month names to numeric (1-12)
    if 'MonthName' in df.columns and df['MonthName'].dtype == 'object':
        month_mapping = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        df['MonthName'] = df['MonthName'].map(month_mapping).fillna(0).astype(int)
    
    # PromoInterval: Convert to numeric (0 = None, 1 = has interval)
    if 'PromoInterval' in df.columns and df['PromoInterval'].dtype == 'object':
        # Convert to binary: 0 if None/empty, 1 if has value
        df['PromoInterval'] = (df['PromoInterval'].notna() & 
                               (df['PromoInterval'] != 'None') & 
                               (df['PromoInterval'] != '')).astype(int)
    
    # Ensure all remaining object columns are converted to numeric
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Date':  # Keep Date as datetime for now
            try:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                # If conversion fails, use label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                mask = df[col].notna()
                if mask.sum() > 0:
                    df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
                    df.loc[~mask, col] = 0
                else:
                    df[col] = 0
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    # Before returning, ensure all columns except Date are numeric
    for col in df.columns:
        if col != 'Date' and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def align_features_with_model(
    df: pd.DataFrame,
    model_metadata: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Align features with model's expected feature order and names.
    
    Args:
        df: DataFrame with features
        model_metadata: Model metadata with expected feature names
    
    Returns:
        Aligned DataFrame with only numeric features
    """
    if model_metadata and 'feature_names' in model_metadata:
        expected_features = model_metadata['feature_names']
        
        # Create a new DataFrame with expected features
        aligned_df = pd.DataFrame(index=df.index)
        
        for feat in expected_features:
            if feat in df.columns:
                aligned_df[feat] = df[feat]
            else:
                # Feature missing - set to 0
                aligned_df[feat] = 0
        
        # Ensure all columns are numeric (drop Date if present)
        if 'Date' in aligned_df.columns:
            aligned_df = aligned_df.drop(columns=['Date'])
        
        # Convert any remaining object columns to numeric
        for col in aligned_df.select_dtypes(include=['object']).columns:
            aligned_df[col] = pd.to_numeric(aligned_df[col], errors='coerce').fillna(0)
        
        # Ensure all columns are numeric types
        aligned_df = aligned_df.astype(float)
        
        return aligned_df
    
    # If no metadata, still ensure Date is dropped and all numeric
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    
    # Convert any object columns to numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

