"""
辅助函数模块
提供各种工具函数和辅助功能
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
    设置全局随机种子，确保实验的可重复性
    
    Args:
        seed: 随机种子值
        deterministic: 是否使用确定性算法（可能影响性能但确保可重复性）
    """
    logger.info(f"设置全局随机种子: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    if deterministic:
        # 设置PyTorch的确定性算法（可能影响性能但确保可重复性）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 设置环境变量以确保完全确定性
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info("全局随机种子设置完成")

def get_seed_from_config(config: Dict[str, Any]) -> int:
    """
    从配置中获取随机种子
    
    Args:
        config: 配置字典
        
    Returns:
        随机种子值，如果未找到则返回默认值42
    """
    # 尝试从不同位置获取种子
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
    确保实验的可重复性
    
    Args:
        config: 配置字典
        deterministic: 是否使用确定性算法
    """
    seed = get_seed_from_config(config)
    set_global_seed(seed, deterministic)
    
    # 记录配置信息
    logger.info(f"实验配置:")
    logger.info(f"  随机种子: {seed}")
    logger.info(f"  确定性模式: {deterministic}")
    logger.info(f"  设备: {config.get('global', {}).get('device', 'auto')}")


def save_results(results: Dict[str, Any], filepath: str, format: str = "json") -> None:
    """
    保存结果到文件
    
    Args:
        results: 要保存的结果字典
        filepath: 文件路径
        format: 保存格式 ("json", "pickle", "csv")
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
        raise ValueError(f"不支持的格式: {format}")


def load_results(filepath: str, format: str = "json") -> Any:
    """
    从文件加载结果
    
    Args:
        filepath: 文件路径
        format: 文件格式 ("json", "pickle", "csv")
    
    Returns:
        加载的数据
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    if format == "json":
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == "csv":
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"不支持的格式: {format}")


def validate_data(data: Union[pd.DataFrame, np.ndarray], 
                 required_columns: List[str] = None,
                 min_rows: int = 1,
                 max_missing_ratio: float = 0.5) -> bool:
    """
    验证数据有效性
    
    Args:
        data: 输入数据
        required_columns: 必需的列名
        min_rows: 最小行数
        max_missing_ratio: 最大缺失值比例
    
    Returns:
        数据是否有效
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # 检查行数
    if len(data) < min_rows:
        raise ValueError(f"数据行数不足: {len(data)} < {min_rows}")
    
    # 检查必需列
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")
    
    # 检查缺失值
    missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
    if missing_ratio > max_missing_ratio:
        raise ValueError(f"缺失值比例过高: {missing_ratio:.2f} > {max_missing_ratio}")
    
    return True


def format_output(data: Any, output_format: str = "dict") -> Any:
    """
    格式化输出
    
    Args:
        data: 输入数据
        output_format: 输出格式 ("dict", "dataframe", "array")
    
    Returns:
        格式化后的数据
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
        raise ValueError(f"不支持的输出格式: {output_format}")


def create_logger(name: str, level: str = "INFO", 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    创建日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
    
    Returns:
        日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory(path: str) -> None:
    """
    确保目录存在
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_extension(filepath: str) -> str:
    """
    获取文件扩展名
    
    Args:
        filepath: 文件路径
    
    Returns:
        文件扩展名
    """
    return Path(filepath).suffix.lower()


def is_numeric_data(data: Union[pd.DataFrame, np.ndarray]) -> bool:
    """
    检查数据是否为数值型
    
    Args:
        data: 输入数据
    
    Returns:
        是否为数值型数据
    """
    if isinstance(data, np.ndarray):
        return np.issubdtype(data.dtype, np.number)
    elif isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]
    else:
        return False


def convert_to_numeric(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """
    将数据转换为数值型
    
    Args:
        data: 输入数据
    
    Returns:
        转换后的数值型数据
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