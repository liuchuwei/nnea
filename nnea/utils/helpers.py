"""
Helper Functions Module
Provides various utility functions and helper functionality
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
import logging

import random
import torch

logger = logging.getLogger(__name__)

def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set global random seed to ensure experiment reproducibility
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (may affect performance but ensures reproducibility)
    """
    logger.info(f"Setting global random seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU case
    
    if deterministic:
        # Set PyTorch deterministic algorithms (may affect performance but ensures reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables to ensure complete determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info("Global random seed setup completed")

def get_seed_from_config(config: Dict[str, Any]) -> int:
    """
    Get random seed from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Random seed value, returns default value 42 if not found
    """
    # Try to get seed from different locations
    seed = config.get('global', {}).get('seed', None)
    if seed is None:
        seed = config.get('seed', None)
    if seed is None:
        seed = config.get('dataset', {}).get('random_state', None)
    if seed is None:
        seed = config.get('training', {}).get('random_state', None)
    
    return seed if seed is not None else 42

def ensure_reproducibility(config: Dict[str, Any], deterministic: bool = True) -> None:
    """
    Ensure experiment reproducibility
    
    Args:
        config: Configuration dictionary
        deterministic: Whether to use deterministic algorithms
    """
    seed = get_seed_from_config(config)
    set_global_seed(seed, deterministic)
    
    # Log configuration information
    logger.info(f"Experiment configuration:")
    logger.info(f"  Random seed: {seed}")
    logger.info(f"  Deterministic mode: {deterministic}")
    logger.info(f"  Device: {config.get('global', {}).get('device', 'auto')}")


def save_results(results: Dict[str, Any], filepath: str, format: str = "json") -> None:
    """
    Save results to file
    
    Args:
        results: Results dictionary to save
        filepath: File path
        format: Save format ("json", "pickle", "csv")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == "json":
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    elif format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif format == "csv":
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
        else:
            pd.DataFrame(results).to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filepath: str, format: str = "json") -> Any:
    """
    Load results from file
    
    Args:
        filepath: File path
        format: File format ("json", "pickle", "csv")
    
    Returns:
        Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")
    
    if format == "json":
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == "csv":
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_data(data: Union[pd.DataFrame, np.ndarray], 
                 required_columns: List[str] = None,
                 min_rows: int = 1,
                 max_missing_ratio: float = 0.5) -> bool:
    """
    Validate data validity
    
    Args:
        data: Input data
        required_columns: Required column names
        min_rows: Minimum number of rows
        max_missing_ratio: Maximum missing value ratio
    
    Returns:
        Whether the data is valid
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # Check number of rows
    if len(data) < min_rows:
        raise ValueError(f"Insufficient data rows: {len(data)} < {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check missing values
    missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
    if missing_ratio > max_missing_ratio:
        raise ValueError(f"Missing value ratio too high: {missing_ratio:.2f} > {max_missing_ratio}")
    
    return True


def format_output(data: Any, output_format: str = "dict") -> Any:
    """
    Format output
    
    Args:
        data: Input data
        output_format: Output format ("dict", "dataframe", "array")
    
    Returns:
        Formatted data
    """
    if output_format == "dict":
        if isinstance(data, pd.DataFrame):
            return data.to_dict()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    elif output_format == "dataframe":
        if isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            return data
    elif output_format == "array":
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, dict):
            return np.array(list(data.values()))
        else:
            return np.array(data)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def create_logger(name: str, level: str = "INFO", 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Create logger
    
    Args:
        name: Logger name
        level: Log level
        log_file: Log file path
    
    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory(path: str) -> None:
    """
    Ensure directory exists
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_extension(filepath: str) -> str:
    """
    Get file extension
    
    Args:
        filepath: File path
    
    Returns:
        File extension
    """
    return Path(filepath).suffix.lower()


def is_numeric_data(data: Union[pd.DataFrame, np.ndarray]) -> bool:
    """
    Check if data is numeric
    
    Args:
        data: Input data
    
    Returns:
        Whether the data is numeric
    """
    if isinstance(data, np.ndarray):
        return np.issubdtype(data.dtype, np.number)
    elif isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]
    else:
        return False


def convert_to_numeric(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """
    Convert data to numeric type
    
    Args:
        data: Input data
    
    Returns:
        Converted numeric data
    """
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number])
    elif isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number):
            return data
        else:
            return data.astype(float)
    else:
        return np.array(data, dtype=float) 