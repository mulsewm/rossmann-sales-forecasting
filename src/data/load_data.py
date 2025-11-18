"""
Module for loading Rossmann sales data from various sources.

This module provides functions to download data from Kaggle and load
CSV files into pandas DataFrames with proper validation.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import logger, DATA_DIR

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_data(
    competition_name: str = "rossmann-store-sales",
    data_dir: Optional[Path] = None
) -> bool:
    """
    Download Rossmann dataset from Kaggle using Kaggle API.
    
    Prerequisites:
        - Kaggle API installed (pip install kaggle)
        - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    
    Args:
        competition_name: Name of the Kaggle competition
        data_dir: Directory to save downloaded files (default: data/raw/)
    
    Returns:
        True if download successful, False otherwise
    
    Raises:
        RuntimeError: If Kaggle API is not configured properly
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    
    try:
        import kaggle
        
        logger.info(f"Downloading data from Kaggle competition: {competition_name}")
        logger.info(f"Saving to: {data_dir}")
        
        # Download all competition files
        kaggle.api.competition_download_files(
            competition=competition_name,
            path=str(data_dir),
            quiet=False
        )
        
        logger.info("Download completed successfully")
        
        # Check if files need to be unzipped
        zip_file = data_dir / f"{competition_name}.zip"
        if zip_file.exists():
            import zipfile
            logger.info("Extracting files...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info("Extraction completed")
            
        return True
        
    except ImportError:
        logger.error("Kaggle library not installed. Install with: pip install kaggle")
        raise RuntimeError("Kaggle library not found")
    
    except Exception as e:
        logger.error(f"Error downloading data from Kaggle: {e}")
        logger.error("Make sure Kaggle API credentials are configured properly")
        logger.error("See: https://github.com/Kaggle/kaggle-api#api-credentials")
        return False


def load_train_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load training data from CSV file.
    
    Args:
        file_path: Path to train.csv file (default: data/raw/train.csv)
    
    Returns:
        DataFrame containing training data
    
    Raises:
        FileNotFoundError: If train.csv is not found
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "train.csv"
    
    logger.info(f"Loading training data from: {file_path}")
    
    if not file_path.exists():
        logger.error(f"Training file not found: {file_path}")
        raise FileNotFoundError(
            f"train.csv not found at {file_path}. "
            "Run download_kaggle_data() first."
        )
    
    # Load data with appropriate dtypes
    dtype_dict = {
        'Store': 'int32',
        'DayOfWeek': 'int8',
        'Sales': 'int32',
        'Customers': 'int32',
        'Open': 'float16',  # Has NaN values
        'Promo': 'int8',
        'StateHoliday': 'object',
        'SchoolHoliday': 'int8'
    }
    
    df = pd.read_csv(
        file_path,
        dtype=dtype_dict,
        parse_dates=['Date'],
        low_memory=False
    )
    
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def load_test_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load test data from CSV file.
    
    Args:
        file_path: Path to test.csv file (default: data/raw/test.csv)
    
    Returns:
        DataFrame containing test data
    
    Raises:
        FileNotFoundError: If test.csv is not found
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "test.csv"
    
    logger.info(f"Loading test data from: {file_path}")
    
    if not file_path.exists():
        logger.error(f"Test file not found: {file_path}")
        raise FileNotFoundError(
            f"test.csv not found at {file_path}. "
            "Run download_kaggle_data() first."
        )
    
    dtype_dict = {
        'Store': 'int32',
        'DayOfWeek': 'int8',
        'Open': 'float16',
        'Promo': 'int8',
        'StateHoliday': 'object',
        'SchoolHoliday': 'int8'
    }
    
    df = pd.read_csv(
        file_path,
        dtype=dtype_dict,
        parse_dates=['Date'],
        low_memory=False
    )
    
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    return df


def load_store_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load store information data from CSV file.
    
    Args:
        file_path: Path to store.csv file (default: data/raw/store.csv)
    
    Returns:
        DataFrame containing store information
    
    Raises:
        FileNotFoundError: If store.csv is not found
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "store.csv"
    
    logger.info(f"Loading store data from: {file_path}")
    
    if not file_path.exists():
        logger.error(f"Store file not found: {file_path}")
        raise FileNotFoundError(
            f"store.csv not found at {file_path}. "
            "Run download_kaggle_data() first."
        )
    
    dtype_dict = {
        'Store': 'int32',
        'StoreType': 'object',
        'Assortment': 'object',
        'CompetitionDistance': 'float32',
        'CompetitionOpenSinceMonth': 'float16',
        'CompetitionOpenSinceYear': 'float16',
        'Promo2': 'int8',
        'Promo2SinceWeek': 'float16',
        'Promo2SinceYear': 'float16',
        'PromoInterval': 'object'
    }
    
    df = pd.read_csv(file_path, dtype=dtype_dict)
    
    logger.info(f"Loaded {len(df):,} stores with {len(df.columns)} features")
    
    return df


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets (train, test, and store).
    
    Returns:
        Tuple of (train_df, test_df, store_df)
    
    Example:
        >>> train, test, store = load_all_data()
        >>> print(f"Training samples: {len(train)}")
    """
    logger.info("Loading all datasets...")
    
    train_df = load_train_data()
    test_df = load_test_data()
    store_df = load_store_data()
    
    logger.info("All datasets loaded successfully")
    
    return train_df, test_df, store_df


def validate_data(df: pd.DataFrame, data_type: str = "train") -> bool:
    """
    Validate data integrity and structure.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data ("train", "test", or "store")
    
    Returns:
        True if validation passes, False otherwise
    """
    logger.info(f"Validating {data_type} data...")
    
    required_columns = {
        "train": ["Store", "Date", "Sales", "Customers", "Open", "Promo"],
        "test": ["Store", "Date", "Open", "Promo"],
        "store": ["Store", "StoreType", "Assortment"]
    }
    
    if data_type not in required_columns:
        logger.error(f"Unknown data type: {data_type}")
        return False
    
    # Check required columns
    missing_cols = set(required_columns[data_type]) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for empty dataframe
    if len(df) == 0:
        logger.error("DataFrame is empty")
        return False
    
    # Check Store IDs
    if df['Store'].isna().any():
        logger.error("Store column contains missing values")
        return False
    
    logger.info("Validation passed")
    return True


# TODO: Add function to merge train/test data with store data
# TODO: Add function to save processed data
# TODO: Add function to load specific date ranges

