"""
Rossmann Sales Forecasting Package

This package contains modules for data processing, feature engineering,
model training, and visualization for the Rossmann sales forecasting project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

__version__ = "0.1.0"
__author__ = "Your Name"

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(
    level: int = logging.INFO,
    log_file: str = None,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: logging.INFO)
        log_file: Path to log file (default: logs/rossmann_{timestamp}.log)
        log_to_console: Whether to output logs to console (default: True)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("rossmann")
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"rossmann_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# Initialize default logger
logger = setup_logging()

# Export commonly used items
__all__ = [
    "setup_logging",
    "logger",
    "PROJECT_ROOT",
    "LOGS_DIR",
    "MODELS_DIR",
    "DATA_DIR",
]

