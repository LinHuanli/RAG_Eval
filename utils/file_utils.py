import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class FileOperationError(Exception):
    """Exception raised for file operation errors."""
    pass


def ensure_directory(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Raises:
        FileOperationError: If directory creation fails
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        error_msg = f"Failed to create directory {directory_path}: {str(e)}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileOperationError: If file loading or parsing fails
    """
    try:
        logger.debug(f"Loading JSON from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in {file_path}: {str(e)}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)
    except Exception as e:
        error_msg = f"Error loading {file_path}: {str(e)}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)


def save_json(data: Any, file_path: str, ensure_dir: bool = True) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        ensure_dir: Whether to ensure the directory exists
        
    Raises:
        FileOperationError: If file saving fails
    """
    try:
        if ensure_dir:
            ensure_directory(os.path.dirname(file_path))
        
        logger.debug(f"Saving JSON to {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        error_msg = f"Error saving to {file_path}: {str(e)}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)


class DirectoryManager:
    """
    Manages directory structure for the RAG evaluation system.
    Provides standardized paths for different types of data and outputs.
    """
    
    # Root directories
    DATA_DIR = "data"
    OUTPUTS_DIR = "outputs"
    LOGS_DIR = "logs"
    
    # Data subdirectories
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    QA_PAIRS_DIR = os.path.join(DATA_DIR, "qa_pairs")
    
    # Output subdirectories
    RUNS_DIR = os.path.join(OUTPUTS_DIR, "runs")
    REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
    
    @classmethod
    def get_data_path(cls, filename: str, data_type: str = "processed") -> str:
        """
        Get path for a data file of specified type.
        
        Args:
            filename: Name of the file
            data_type: Type of data ('raw', 'processed', or 'qa_pairs')
            
        Returns:
            Path to the file
        """
        if data_type == "raw":
            directory = cls.RAW_DATA_DIR
        elif data_type == "processed":
            directory = cls.PROCESSED_DATA_DIR
        elif data_type == "qa_pairs":
            directory = cls.QA_PAIRS_DIR
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        
        ensure_directory(directory)
        return os.path.join(directory, filename)
    
    @classmethod
    def get_run_dir(cls, run_name: Optional[str] = None) -> str:
        """
        Get path for a run directory.
        
        Args:
            run_name: Name of the run (if None, generates timestamped name)
            
        Returns:
            Path to the run directory
        """
        if run_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        run_dir = os.path.join(cls.RUNS_DIR, run_name)
        ensure_directory(run_dir)
        return run_dir
    
    @classmethod
    def get_log_path(cls, log_name: str = "rag_eval.log") -> str:
        """
        Get path for a log file.
        
        Args:
            log_name: Name of the log file
            
        Returns:
            Path to the log file
        """
        ensure_directory(cls.LOGS_DIR)
        return os.path.join(cls.LOGS_DIR, log_name)
    
    @classmethod
    def get_report_path(cls, report_name: str) -> str:
        """
        Get path for a report file.
        
        Args:
            report_name: Name of the report file
            
        Returns:
            Path to the report file
        """
        ensure_directory(cls.REPORTS_DIR)
        return os.path.join(cls.REPORTS_DIR, report_name)


def generate_output_path(base_dir: str = None, prefix: str = "output", 
                         suffix: str = ".json", run_name: Optional[str] = None) -> str:
    """
    Generate a standardized output file path.
    
    Args:
        base_dir: Base directory (if None, uses DirectoryManager.get_run_dir())
        prefix: File name prefix
        suffix: File extension
        run_name: Name of the run (if None, generates timestamped name)
        
    Returns:
        Generated file path
        
    Raises:
        FileOperationError: If directory creation fails
    """
    try:
        # Use either provided base_dir or get a run directory
        directory = base_dir if base_dir else DirectoryManager.get_run_dir(run_name)
        ensure_directory(directory)
        
        # Generate filename with timestamp if no run_name is provided
        if not run_name and not base_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}{suffix}"
        else:
            filename = f"{prefix}{suffix}"
            
        return os.path.join(directory, filename)
    except Exception as e:
        raise FileOperationError(f"Failed to create output path: {str(e)}")