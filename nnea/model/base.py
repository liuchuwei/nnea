"""
NNEA模型基类
提供统一的模型接口和基础功能
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    NNEA模型基类
    所有NNEA模型都应该继承此类
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            config: 模型配置字典
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.device = torch.device(config.get('device', 'cpu'))
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def build(self, nadata) -> None:
        """
        构建模型
        
        Args:
            nadata: nadata对象，包含数据和配置信息
        """
        pass
    
    @abstractmethod
    def train(self, nadata, **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            nadata: nadata对象
            **kwargs: 额外参数
            
        Returns:
            训练结果字典
        """
        pass
    
    @abstractmethod
    def predict(self, nadata) -> np.ndarray:
        """
        模型预测
        
        Args:
            nadata: nadata对象
            
        Returns:
            预测结果
        """
        pass
    
    @abstractmethod
    def evaluate(self, nadata, split='test') -> Dict[str, float]:
        """
        模型评估
        
        Args:
            nadata: nadata对象
            split: 评估的数据集分割
            
        Returns:
            评估指标字典
        """
        pass
    
    @abstractmethod
    def explain(self, nadata, method='importance') -> Dict[str, Any]:
        """
        模型解释
        
        Args:
            nadata: nadata对象
            method: 解释方法
            
        Returns:
            解释结果字典
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未构建")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        加载模型
        
        Args:
            filepath: 加载路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.config = checkpoint['config']
        self.is_trained = checkpoint['is_trained']
        
        # 重新构建模型
        self.build(None)  # 这里需要传入nadata，但加载时可能没有
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"模型已从 {filepath} 加载")
    
    def to_device(self, device: str) -> None:
        """
        将模型移动到指定设备
        
        Args:
            device: 设备名称
        """
        if self.model is not None:
            self.device = torch.device(device)
            self.model.to(self.device)
            self.logger.info(f"模型已移动到设备: {device}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要
        
        Returns:
            模型摘要字典
        """
        summary = {
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'device': str(self.device),
            'config': self.config
        }
        return summary 