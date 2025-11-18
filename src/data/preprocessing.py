"""
Data preprocessing and cleaning utilities for Rossmann sales data.

This module provides functions for cleaning, transforming, and preparing
the data for modeling.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import logger


def clean_data(df: pd.DataFrame, data_type: str = "train") -> pd.DataFrame:
    """
    Clean and preprocess the dataset.
    
    Args:
        df: Input DataFrame
        data_type: Type of data ("train", "test", or "store")
    
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning {data_type} data...")
    logger.info(f"Initial shape: {df.shape}")
    
    df_clean = df.copy()
    
    # Convert Date to datetime if not already
    if 'Date' in df_clean.columns and df_clean['Date'].dtype != 'datetime64[ns]':
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    # Sort by Store and Date
    if 'Store' in df_clean.columns and 'Date' in df_clean.columns:
        df_clean = df_clean.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    # Handle StateHoliday (convert 0 to '0' for consistency)
    if 'StateHoliday' in df_clean.columns:
        df_clean['StateHoliday'] = df_clean['StateHoliday'].astype(str)
        df_clean['StateHoliday'] = df_clean['StateHoliday'].replace('0', '0')
    
    # For training data, remove rows where store is closed and sales = 0
    if data_type == "train":
        initial_rows = len(df_clean)
        # Keep rows where store is open OR sales > 0
        df_clean = df_clean[(df_clean['Open'] == 1) | (df_clean['Sales'] > 0)]
        removed_rows = initial_rows - len(df_clean)
        logger.info(f"Removed {removed_rows:,} rows where store was closed and sales = 0")
    
    logger.info(f"Final shape: {df_clean.shape}")
    
    return df_clean


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Dictionary mapping column names to imputation strategies
                 Options: 'mean', 'median', 'mode', 'zero', 'forward_fill', 'drop'
    
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values...")
    
    df_imputed = df.copy()
    
    # Default strategies for common missing values
    if strategy is None:
        strategy = {
            'Open': 'mode',
            'CompetitionDistance': 'median',
            'CompetitionOpenSinceMonth': 'mode',
            'CompetitionOpenSinceYear': 'mode',
            'Promo2SinceWeek': 'zero',
            'Promo2SinceYear': 'zero',
            'PromoInterval': 'none'
        }
    
    for column in df_imputed.columns:
        missing_count = df_imputed[column].isna().sum()
        
        if missing_count > 0:
            logger.info(f"Column '{column}': {missing_count} missing values")
            
            if column in strategy:
                method = strategy[column]
                
                if method == 'mean':
                    df_imputed[column].fillna(df_imputed[column].mean(), inplace=True)
                elif method == 'median':
                    df_imputed[column].fillna(df_imputed[column].median(), inplace=True)
                elif method == 'mode':
                    df_imputed[column].fillna(df_imputed[column].mode()[0], inplace=True)
                elif method == 'zero':
                    df_imputed[column].fillna(0, inplace=True)
                elif method == 'forward_fill':
                    df_imputed[column].fillna(method='ffill', inplace=True)
                elif method == 'none':
                    df_imputed[column].fillna('None', inplace=True)
                elif method == 'drop':
                    df_imputed.dropna(subset=[column], inplace=True)
                
                logger.info(f"  Applied strategy: {method}")
    
    remaining_missing = df_imputed.isna().sum().sum()
    logger.info(f"Remaining missing values: {remaining_missing}")
    
    return df_imputed


def encode_categorical_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    encoding_type: str = 'label'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        columns: List of columns to encode (if None, auto-detect categorical)
        encoding_type: Type of encoding ('label', 'onehot', 'target')
    
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    logger.info(f"Encoding categorical features using {encoding_type} encoding...")
    
    df_encoded = df.copy()
    encoders = {}
    
    # Auto-detect categorical columns if not specified
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Encoding columns: {columns}")
    
    for column in columns:
        if column not in df_encoded.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            continue
        
        if encoding_type == 'label':
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            encoders[column] = le
            logger.info(f"  {column}: {len(le.classes_)} unique values")
        
        elif encoding_type == 'onehot':
            dummies = pd.get_dummies(df_encoded[column], prefix=column, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(column, axis=1, inplace=True)
            encoders[column] = list(dummies.columns)
            logger.info(f"  {column}: created {len(dummies.columns)} dummy variables")
    
    return df_encoded, encoders


def merge_store_data(
    df: pd.DataFrame,
    store_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge sales data with store information.
    
    Args:
        df: Train or test DataFrame
        store_df: Store information DataFrame
    
    Returns:
        Merged DataFrame
    """
    logger.info("Merging with store data...")
    
    df_merged = df.merge(store_df, on='Store', how='left')
    
    logger.info(f"Merged shape: {df_merged.shape}")
    
    return df_merged


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from Date column.
    
    Args:
        df: DataFrame with Date column
    
    Returns:
        DataFrame with additional time features
    """
    logger.info("Creating time-based features...")
    
    df_time = df.copy()
    
    if 'Date' not in df_time.columns:
        logger.warning("Date column not found")
        return df_time
    
    # Extract date components
    df_time['Year'] = df_time['Date'].dt.year
    df_time['Month'] = df_time['Date'].dt.month
    df_time['Day'] = df_time['Date'].dt.day
    df_time['Week'] = df_time['Date'].dt.isocalendar().week
    df_time['Quarter'] = df_time['Date'].dt.quarter
    df_time['DayOfYear'] = df_time['Date'].dt.dayofyear
    
    # Is weekend
    df_time['IsWeekend'] = (df_time['DayOfWeek'] >= 6).astype(int)
    
    # Is month start/end
    df_time['IsMonthStart'] = df_time['Date'].dt.is_month_start.astype(int)
    df_time['IsMonthEnd'] = df_time['Date'].dt.is_month_end.astype(int)
    
    # Days until/since specific events (can be extended)
    # TODO: Add days until Christmas, Easter, etc.
    
    logger.info(f"Created {len([c for c in df_time.columns if c not in df.columns])} new features")
    
    return df_time


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate preprocessed data.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if validation passes
    """
    logger.info("Validating preprocessed data...")
    
    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        logger.error("Dataset contains infinite values")
        return False
    
    # Check for excessive missing values
    missing_pct = (df.isna().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > 50]
    
    if len(high_missing) > 0:
        logger.warning(f"Columns with >50% missing values: {list(high_missing.index)}")
    
    logger.info("Validation completed")
    return True


# TODO: Add feature scaling functions
# TODO: Add outlier detection and handling
# TODO: Add feature selection utilities
# TODO: Add data augmentation for imbalanced classes

