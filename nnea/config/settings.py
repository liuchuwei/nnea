"""
配置设置模块
提供配置文件的加载、保存和验证功能
"""

import os
import json
import toml
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


def load_config(filepath: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        filepath: 配置文件路径
    
    Returns:
        配置字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"配置文件不存在: {filepath}")
    
    file_ext = Path(filepath).suffix.lower()
    
    if file_ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_ext == '.toml':
        return toml.load(filepath)
    elif file_ext in ['.yml', '.yaml']:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {file_ext}")


def save_config(config: Dict[str, Any], filepath: str, format: str = "json") -> None:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        filepath: 文件路径
        format: 保存格式 ("json", "toml", "yaml")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == "json":
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    elif format == "toml":
        with open(filepath, 'w', encoding='utf-8') as f:
            toml.dump(config, f)
    elif format == "yaml":
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"不支持的格式: {format}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新配置
    
    Args:
        config: 原始配置
        updates: 更新内容
    
    Returns:
        更新后的配置
    """
    def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    return deep_update(config.copy(), updates)


def validate_config(config: Dict[str, Any], required_keys: list = None) -> bool:
    """
    验证配置有效性
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
    
    Returns:
        配置是否有效
    """
    if required_keys:
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            raise ValueError(f"缺少必需的配置项: {missing_keys}")
    
    # 验证数据类型
    if 'model' in config:
        if not isinstance(config['model'], dict):
            raise ValueError("model配置必须是字典类型")
    
    if 'data' in config:
        if not isinstance(config['data'], dict):
            raise ValueError("data配置必须是字典类型")
    
    if 'training' in config:
        if not isinstance(config['training'], dict):
            raise ValueError("training配置必须是字典类型")
    
    return True


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    获取嵌套配置值
    
    Args:
        config: 配置字典
        key_path: 键路径 (用.分隔)
        default: 默认值
    
    Returns:
        配置值
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    设置嵌套配置值
    
    Args:
        config: 配置字典
        key_path: 键路径 (用.分隔)
        value: 要设置的值
    
    Returns:
        更新后的配置
    """
    keys = key_path.split('.')
    config_copy = config.copy()
    current = config_copy
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return config_copy


def merge_configs(configs: list) -> Dict[str, Any]:
    """
    合并多个配置
    
    Args:
        configs: 配置列表
    
    Returns:
        合并后的配置
    """
    merged = {}
    for config in configs:
        merged = update_config(merged, config)
    return merged


def create_config_template() -> Dict[str, Any]:
    """
    创建配置模板
    
    Returns:
        配置模板
    """
    return {
        "model": {
            "type": "neural_network",
            "layers": [100, 50, 25],
            "activation": "relu",
            "dropout": 0.2
        },
        "data": {
            "train_path": "",
            "test_path": "",
            "validation_split": 0.2,
            "batch_size": 32
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy"
        },
        "output": {
            "model_path": "models/",
            "results_path": "results/",
            "log_path": "logs/"
        }
    } 