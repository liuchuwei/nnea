import logging
import coloredlogs
import os
from datetime import datetime

def setup_logging(level='INFO', log_file=None, experiment_name=None):
    """
    设置简洁现代风格的日志配置
    
    Args:
        level: 日志级别，默认为INFO
        log_file: 日志文件路径，如果为None则使用默认路径
        experiment_name: 实验名称，用于生成日志文件名
    """
    fmt = '%(asctime)s | %(levelname)s | %(message)s'
    datefmt = '%H:%M:%S'
    
    # 如果提供了log_file，使用指定的路径
    if log_file is not None:
        # 确保日志文件的目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    else:
        # 创建默认日志目录
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
        else:
            log_file = f"{log_dir}/experiment_{timestamp}.log"
    
    # 基础配置
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(log_file, mode='w', encoding='utf-8')  # 文件输出
        ],
        force=True
    )
    
    # 彩色日志配置
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
    获取logger实例
    
    Args:
        name: logger名称，默认为None
    
    Returns:
        logger实例
    """
    return logging.getLogger(name)

# 自动设置日志配置
setup_logging()

# 获取默认logger
logger = get_logger(__name__)
