import logging
import coloredlogs
import os
from datetime import datetime

def setup_logging(level='INFO', log_file=None, experiment_name=None):
    """
    Set up concise modern style logging configuration
    
    Args:
        level: Log level, default is INFO
        log_file: Log file path, if None, use default path
        experiment_name: Experiment name, used to generate log file name
    """
    fmt = '%(asctime)s | %(levelname)s | %(message)s'
    datefmt = '%H:%M:%S'
    
    # If log_file is provided, use the specified path
    if log_file is not None:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    else:
        # Create default log directory
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
        else:
            log_file = f"{log_dir}/experiment_{timestamp}.log"
    
    # Basic configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file, mode='w', encoding='utf-8')  # File output
        ],
        force=True
    )
    
    # Colored logs configuration
    coloredlogs.install(
        level=level,
        fmt=fmt,
        datefmt=datefmt,
        field_styles={
            'levelname': {'color': 'white', 'bold': True},
            'asctime': {'color': 'cyan'}
        },
        level_styles={
            'debug': {'color': 'blue'},
            'info': {'color': 'green'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red'},
            'critical': {'color': 'red', 'bold': True}
        }
    )
    
    return log_file

def get_logger(name=None):
    """
    Get logger instance
    
    Args:
        name: Logger name, default is None
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

# Automatically set up logging configuration
setup_logging()

# Get default logger
logger = get_logger(__name__)
