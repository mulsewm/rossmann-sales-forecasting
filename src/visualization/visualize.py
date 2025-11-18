"""
Visualization functions for Rossmann sales forecasting project.

This module provides various plotting functions for:
- Exploratory Data Analysis (EDA)
- Feature analysis
- Model performance evaluation
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_sales_over_time(
    df: pd.DataFrame,
    date_col: str = 'Date',
    sales_col: str = 'Sales',
    store_id: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 6),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot sales over time.
    
    Args:
        df: DataFrame with date and sales columns
        date_col: Name of date column
        sales_col: Name of sales column
        store_id: Optional store ID to filter
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    logger.info("Plotting sales over time...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_df = df.copy()
    if store_id is not None:
        plot_df = plot_df[plot_df['Store'] == store_id]
        title = f'Sales Over Time - Store {store_id}'
    else:
        # Aggregate by date if multiple stores
        plot_df = plot_df.groupby(date_col)[sales_col].sum().reset_index()
        title = 'Total Sales Over Time (All Stores)'
    
    ax.plot(plot_df[date_col], plot_df[sales_col], linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_sales_distribution(
    df: pd.DataFrame,
    sales_col: str = 'Sales',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot sales distribution with histogram and box plot.
    
    Args:
        df: DataFrame with sales column
        sales_col: Name of sales column
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    logger.info("Plotting sales distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(df[sales_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Sales')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Sales Distribution')
    axes[0].axvline(df[sales_col].mean(), color='red', linestyle='--', label='Mean')
    axes[0].axvline(df[sales_col].median(), color='green', linestyle='--', label='Median')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(df[sales_col])
    axes[1].set_ylabel('Sales')
    axes[1].set_title('Sales Box Plot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    logger.info(f"Plotting top {top_n} feature importances...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_df = feature_importance_df.head(top_n).sort_values('importance')
    
    ax.barh(plot_df['feature'], plot_df['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    logger.info("Plotting predictions vs actual...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Sales')
    ax.set_ylabel('Predicted Sales')
    ax.set_title('Predicted vs Actual Sales')
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot residuals analysis.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    logger.info("Plotting residuals...")
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Sales')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    
    # Residuals distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].axvline(0, color='r', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot correlation heatmap.
    
    Args:
        df: DataFrame
        columns: Specific columns to include (if None, use all numeric)
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    logger.info("Plotting correlation heatmap...")
    
    if columns is None:
        corr_df = df.select_dtypes(include=[np.number])
    else:
        corr_df = df[columns]
    
    correlation = corr_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        correlation,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_sales_by_category(
    df: pd.DataFrame,
    category_col: str,
    sales_col: str = 'Sales',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot sales by categorical variable.
    
    Args:
        df: DataFrame
        category_col: Categorical column name
        sales_col: Sales column name
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    logger.info(f"Plotting sales by {category_col}...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df.groupby(category_col)[sales_col].mean().sort_values().plot(kind='bar', ax=ax)
    ax.set_xlabel(category_col)
    ax.set_ylabel(f'Average {sales_col}')
    ax.set_title(f'Average {sales_col} by {category_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def create_eda_report(
    df: pd.DataFrame,
    output_dir: Path,
    sample_stores: List[int] = None
) -> None:
    """
    Create a comprehensive EDA report with multiple visualizations.
    
    Args:
        df: DataFrame
        output_dir: Directory to save figures
        sample_stores: List of stores to analyze individually
    """
    logger.info("Creating EDA report...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall sales over time
    plot_sales_over_time(df, save_path=output_dir / 'sales_over_time.png')
    
    # Sales distribution
    plot_sales_distribution(df, save_path=output_dir / 'sales_distribution.png')
    
    # Correlation heatmap (sample of columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]
    plot_correlation_heatmap(df, columns=numeric_cols, 
                            save_path=output_dir / 'correlation_heatmap.png')
    
    # Sales by day of week
    if 'DayOfWeek' in df.columns:
        plot_sales_by_category(df, 'DayOfWeek', 
                              save_path=output_dir / 'sales_by_dayofweek.png')
    
    # Sample stores analysis
    if sample_stores and 'Store' in df.columns:
        for store_id in sample_stores[:3]:  # Limit to 3 stores
            plot_sales_over_time(df, store_id=store_id,
                               save_path=output_dir / f'store_{store_id}_sales.png')
    
    logger.info(f"EDA report saved to: {output_dir}")


# TODO: Add interactive plots using Plotly
# TODO: Add time series decomposition plots
# TODO: Add model comparison visualizations
# TODO: Add geographical visualizations if location data available

