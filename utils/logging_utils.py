import os
import logging
import sys
from typing import Optional

from config.config_manager import get_config


def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses config value)
    """
    config = get_config()
    
    # Get log level from args, config, or default to INFO
    if log_level is None:
        log_level = config.get("logging", "level", default="INFO")
    
    # Get log file from args, config, or default to None (console only)
    if log_file is None:
        log_file = config.get("logging", "file", default=None)
    
    # Create log directory if it doesn't exist and a log file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Set numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Define log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log configuration details
    logging.info(f"Logging initialized with level: {log_level}")
    if log_file:
        logging.info(f"Logging to file: {log_file}")