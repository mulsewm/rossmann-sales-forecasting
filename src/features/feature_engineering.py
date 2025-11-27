"""
Feature engineering functions for Rossmann sales forecasting.

This module creates various features including:
- Lag features (previous sales)
- Rolling statistics (moving averages, std)
- Promotion features
- Competition features
- Calendar features
"""

import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import logger


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'Sales',
    group_cols: List[str] = ['Store'],
    lags: List[int] = [1, 7, 14, 30]
) -> pd.DataFrame:
    """
    Create lag features (previous values) for time series.
    
    Args:
        df: Input DataFrame
        target_col: Column to create lags for
        group_cols: Columns to group by (e.g., Store)
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features
    """
    logger.info(f"Creating lag features for {target_col}...")
    
    df_lag = df.copy()
    
    for lag in lags:
        feature_name = f'{target_col}_Lag{lag}'
        df_lag[feature_name] = df_lag.groupby(group_cols)[target_col].shift(lag)
        logger.info(f"  Created: {feature_name}")
    
    return df_lag


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'Sales',
    group_cols: List[str] = ['Store'],
    windows: List[int] = [7, 14, 30],
    functions: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Args:
        df: Input DataFrame
        target_col: Column to calculate statistics for
        group_cols: Columns to group by
        windows: List of window sizes
        functions: Statistical functions to apply
    
    Returns:
        DataFrame with rolling features
    """
    logger.info(f"Creating rolling features for {target_col}...")
    
    df_roll = df.copy()
    
    for window in windows:
        for func in functions:
            feature_name = f'{target_col}_Rolling{window}_{func}'
            
            if func == 'mean':
                df_roll[feature_name] = df_roll.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
            elif func == 'std':
                df_roll[feature_name] = df_roll.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
            elif func == 'min':
                df_roll[feature_name] = df_roll.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
                )
            elif func == 'max':
                df_roll[feature_name] = df_roll.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
                )
            
            logger.info(f"  Created: {feature_name}")
    
    return df_roll


def create_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to promotions.
    
    Args:
        df: Input DataFrame with Promo columns
    
    Returns:
        DataFrame with promotion features
    """
    logger.info("Creating promotion features...")
    
    df_promo = df.copy()
    
    # Check if store is in Promo2
    if 'Promo2' in df_promo.columns:
        df_promo['IsPromo2Active'] = df_promo['Promo2'].astype(int)
    
    # Days since promo started
    if all(col in df_promo.columns for col in ['Date', 'Promo2SinceYear', 'Promo2SinceWeek']):
        df_promo['Promo2SinceDate'] = pd.to_datetime(
            df_promo['Promo2SinceYear'].astype(str) + '-W' + 
            df_promo['Promo2SinceWeek'].astype(str) + '-1',
            format='%Y-W%W-%w',
            errors='coerce'
        )
        df_promo['DaysSincePromo2'] = (df_promo['Date'] - df_promo['Promo2SinceDate']).dt.days
        df_promo['DaysSincePromo2'] = df_promo['DaysSincePromo2'].fillna(0)
        logger.info("  Created: DaysSincePromo2")
    
    # Is current month in PromoInterval
    if all(col in df_promo.columns for col in ['Month', 'PromoInterval']):
        month_mapping = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
            9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        df_promo['MonthName'] = df_promo['Month'].map(month_mapping)
        df_promo['IsPromoMonth'] = df_promo.apply(
            lambda row: 1 if pd.notna(row['PromoInterval']) and 
                        row['MonthName'] in row['PromoInterval'] else 0,
            axis=1
        )
        logger.info("  Created: IsPromoMonth")
    
    # Promo on both channels
    if all(col in df_promo.columns for col in ['Promo', 'Promo2']):
        df_promo['PromoOnBoth'] = (df_promo['Promo'] & df_promo['Promo2']).astype(int)
        logger.info("  Created: PromoOnBoth")
    
    return df_promo


def create_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to competition.
    
    Args:
        df: Input DataFrame with competition columns
    
    Returns:
        DataFrame with competition features
    """
    logger.info("Creating competition features...")
    
    df_comp = df.copy()
    
    # Has competition
    if 'CompetitionDistance' in df_comp.columns:
        df_comp['HasCompetition'] = df_comp['CompetitionDistance'].notna().astype(int)
        
        # Competition distance bins
        df_comp['CompDistanceBin'] = pd.cut(
            df_comp['CompetitionDistance'],
            bins=[0, 500, 1000, 2000, 5000, np.inf],
            labels=['VeryClose', 'Close', 'Medium', 'Far', 'VeryFar']
        )
        logger.info("  Created: HasCompetition, CompDistanceBin")
    
    # Months since competition opened
    if all(col in df_comp.columns for col in ['Date', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']):
        df_comp['CompetitionOpenDate'] = pd.to_datetime(
            df_comp['CompetitionOpenSinceYear'].astype(str) + '-' + 
            df_comp['CompetitionOpenSinceMonth'].astype(str) + '-01',
            format='%Y-%m-%d',
            errors='coerce'
        )
        # Calculate months difference using a supported method
        # Use year and month difference for accurate calculation
        def calculate_months_diff(row):
            if pd.isna(row['CompetitionOpenDate']) or pd.isna(row['Date']):
                return 0
            try:
                years_diff = row['Date'].year - row['CompetitionOpenDate'].year
                months_diff = row['Date'].month - row['CompetitionOpenDate'].month
                return years_diff * 12 + months_diff
            except:
                return 0
        
        df_comp['MonthsSinceCompetition'] = df_comp.apply(calculate_months_diff, axis=1).fillna(0)
        logger.info("  Created: MonthsSinceCompetition")
    
    return df_comp


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to customers.
    
    Args:
        df: Input DataFrame with Customers column
    
    Returns:
        DataFrame with customer features
    """
    logger.info("Creating customer features...")
    
    df_cust = df.copy()
    
    if 'Customers' in df_cust.columns and 'Sales' in df_cust.columns:
        # Average sales per customer
        df_cust['SalesPerCustomer'] = df_cust['Sales'] / (df_cust['Customers'] + 1)
        logger.info("  Created: SalesPerCustomer")
    
    return df_cust


def create_store_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated store-level features.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with store features
    """
    logger.info("Creating store-level features...")
    
    df_store = df.copy()
    
    # TODO: Add store-level aggregations
    # - Average sales per store
    # - Sales volatility per store
    # - Trend features per store
    
    return df_store


def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create holiday-related features.
    
    Args:
        df: Input DataFrame with holiday columns
    
    Returns:
        DataFrame with holiday features
    """
    logger.info("Creating holiday features...")
    
    df_holiday = df.copy()
    
    # Is any kind of holiday
    if all(col in df_holiday.columns for col in ['StateHoliday', 'SchoolHoliday']):
        df_holiday['IsHoliday'] = (
            (df_holiday['StateHoliday'] != '0') | 
            (df_holiday['SchoolHoliday'] == 1)
        ).astype(int)
        logger.info("  Created: IsHoliday")
    
    # Days before/after holiday (requires sorting by date)
    # TODO: Implement days_to_holiday and days_from_holiday
    
    return df_holiday


def create_all_features(
    df: pd.DataFrame,
    include_lag: bool = True,
    include_rolling: bool = True,
    include_promo: bool = True,
    include_competition: bool = True
) -> pd.DataFrame:
    """
    Create all feature sets.
    
    Args:
        df: Input DataFrame
        include_lag: Whether to include lag features
        include_rolling: Whether to include rolling features
        include_promo: Whether to include promotion features
        include_competition: Whether to include competition features
    
    Returns:
        DataFrame with all features
    """
    logger.info("Creating all features...")
    
    df_features = df.copy()
    
    # Time-based features (from preprocessing)
    from src.data.preprocessing import create_time_features
    df_features = create_time_features(df_features)
    
    # Promotion features
    if include_promo:
        df_features = create_promotion_features(df_features)
    
    # Competition features
    if include_competition:
        df_features = create_competition_features(df_features)
    
    # Holiday features
    df_features = create_holiday_features(df_features)
    
    # Customer features
    if 'Customers' in df_features.columns:
        df_features = create_customer_features(df_features)
    
    # Lag features (only for training data with Sales column)
    if include_lag and 'Sales' in df_features.columns:
        df_features = create_lag_features(df_features)
    
    # Rolling features (only for training data)
    if include_rolling and 'Sales' in df_features.columns:
        df_features = create_rolling_features(df_features)
    
    logger.info(f"Total features created: {df_features.shape[1]}")
    
    return df_features


# TODO: Add feature importance analysis
# TODO: Add feature selection methods
# TODO: Add interaction features
# TODO: Add polynomial features for specific variables

