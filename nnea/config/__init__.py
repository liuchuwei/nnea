"""
配置模块
提供配置管理和参数设置功能
"""

from .settings import *
from .defaults import *

__all__ = [
    "load_config",
    "save_config", 
    "get_default_config",
    "update_config",
    "validate_config"
] 