"""
Configuration Settings Module
Provides configuration file loading, saving and validation functionality
"""

import os
import json
import toml
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration file
    
    Args:
        filepath: Configuration file path
    
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file does not exist: {filepath}")
    
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
        raise ValueError(f"Unsupported configuration file format: {file_ext}")


def save_config(config: Dict[str, Any], filepath: str, format: str = "json") -> None:
    """
    Save configuration file
    
    Args:
        config: Configuration dictionary
        filepath: File path
        format: Save format ("json", "toml", "yaml")
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
        raise ValueError(f"Unsupported format: {format}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration
    
    Args:
        config: Original configuration
        updates: Update content
    
    Returns:
        Updated configuration
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
    Validate configuration validity
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
    
    Returns:
        Whether the configuration is valid
    """
    if required_keys:
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required configuration items: {missing_keys}")
    
    # Validate data types
    if 'model' in config:
        if not isinstance(config['model'], dict):
            raise ValueError("model configuration must be a dictionary type")
    
    if 'data' in config:
        if not isinstance(config['data'], dict):
            raise ValueError("data configuration must be a dictionary type")
    
    if 'training' in config:
        if not isinstance(config['training'], dict):
            raise ValueError("training configuration must be a dictionary type")
    
    return True


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value
    
    Args:
        config: Configuration dictionary
        key_path: Key path (separated by .)
        default: Default value
    
    Returns:
        Configuration value
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
    Set nested configuration value
    
    Args:
        config: Configuration dictionary
        key_path: Key path (separated by .)
        value: Value to set
    
    Returns:
        Updated configuration
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
    Merge multiple configurations
    
    Args:
        configs: Configuration list
    
    Returns:
        Merged configuration
    """
    merged = {}
    for config in configs:
        merged = update_config(merged, config)
    return merged


def create_config_template() -> Dict[str, Any]:
    """
    Create configuration template
    
    Returns:
        Configuration template
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