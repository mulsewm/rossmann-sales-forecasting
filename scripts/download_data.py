"""
Script to download Rossmann Store Sales data from Kaggle.

Prerequisites:
    - Kaggle API installed: pip install kaggle
    - Kaggle credentials configured: ~/.kaggle/kaggle.json

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src import logger, DATA_DIR
from src.data.load_data import download_kaggle_data, validate_data, load_all_data


def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download Rossmann Store Sales data from Kaggle'
    )
    
    parser.add_argument(
        '--competition',
        type=str,
        default='rossmann-store-sales',
        help='Kaggle competition name (default: rossmann-store-sales)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATA_DIR / 'raw',
        help='Directory to save downloaded data'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate data after download'
    )
    
    return parser


def main():
    """Main function to download and optionally validate data."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Rossmann Data Download Script")
    logger.info("="*60)
    
    # Check if data directory exists
    args.data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {args.data_dir}")
    
    # Download data
    logger.info(f"Downloading data from Kaggle competition: {args.competition}")
    
    try:
        success = download_kaggle_data(
            competition_name=args.competition,
            data_dir=args.data_dir
        )
        
        if not success:
            logger.error("Data download failed")
            sys.exit(1)
        
        logger.info("✓ Data downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error during download: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Install Kaggle API: pip install kaggle")
        logger.error("2. Setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        logger.error("3. Accept competition rules on Kaggle website")
        sys.exit(1)
    
    # Validate data if requested
    if args.validate:
        logger.info("\nValidating downloaded data...")
        
        try:
            train_df, test_df, store_df = load_all_data()
            
            # Validate each dataset
            train_valid = validate_data(train_df, data_type='train')
            test_valid = validate_data(test_df, data_type='test')
            store_valid = validate_data(store_df, data_type='store')
            
            if train_valid and test_valid and store_valid:
                logger.info("✓ All datasets validated successfully")
                
                # Print summary statistics
                logger.info("\nDataset Summary:")
                logger.info(f"  Training data: {len(train_df):,} rows")
                logger.info(f"  Test data: {len(test_df):,} rows")
                logger.info(f"  Store data: {len(store_df):,} stores")
                logger.info(f"  Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
                logger.info(f"  Unique stores: {train_df['Store'].nunique()}")
                logger.info(f"  Total sales: ${train_df['Sales'].sum():,.0f}")
                
            else:
                logger.error("✗ Data validation failed")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("Download completed successfully!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Explore the data: jupyter lab notebooks/01_data_exploration.ipynb")
    logger.info("2. Run feature engineering: jupyter lab notebooks/02_feature_engineering.ipynb")
    logger.info("3. Train models: python scripts/train_pipeline.py")


if __name__ == "__main__":
    main()

