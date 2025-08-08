"""
NNEA (Neural Network with Explainable Architecture) 模型
实现可解释的神经网络架构，支持基因集学习和生物学解释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import logging

from .base import BaseModel
from .nnea_layers import TrainableGeneSetLayer, BiologicalConstraintLayer
# 修复导入路径问题
import sys
import os
# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
if project_root not in sys.path:
    sys.path.append(project_root)
try:
    from utils.train_utils import AttentionBlock
except ImportError:
    # 如果导入失败，创建一个简单的AttentionBlock类
    import torch.nn as nn
    class AttentionBlock(nn.Module):
        def __init__(self, input_dim):
            super(AttentionBlock, self).__init__()
            self.attention = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            attention_weights = torch.softmax(self.attention(x), dim=1)
            return torch.sum(attention_weights * x, dim=1)
        
        def get_attention_weights(self):
            return self.attention.weight

logger = logging.getLogger(__name__)

class NNEAModel(nn.Module):
    """
    NNEA神经网络模型
    以TrainableGeneSetLayer为核心，包含注意力机制和可解释性组件
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化NNEA模型
        
        Args:
            config: 模型配置
        """
        super(NNEAModel, self).__init__()
        
        # 设备配置
        device_config = config.get('device', 'cpu')
        if device_config == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # 模型参数
        self.input_dim = config.get('input_dim', 0)
        self.output_dim = config.get('output_dim', 1)
        self.dropout_rate = config.get('dropout', 0.3)
        self.activation = config.get('activation', 'relu')
        
        # 任务类型 - 用于判断是否需要输出概率
        self.task_type = config.get('global').get('task', 'classification')
        
        # 获取nnea配置部分
        nnea_config = config.get('nnea', {})
        
        # NNEA特定参数 - 从nnea配置中获取
        self.use_piror_knowledge = nnea_config.get('piror_knowledge', {}).get('use_piror_knowledge', False)
        self.piror_knowledge = nnea_config.get('piror_knowledge', {}).get('piror_knowledge', None)
        self.freeze_piror = nnea_config.get('piror_knowledge', {}).get('freeze_piror', True)
        # 如果有piror_knowledge，则num_genesets等于piror_knowledge的行数，否则取配置
        if self.piror_knowledge is not None:
            self.num_genesets = self.piror_knowledge.shape[0]
        else:
            self.num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
        
        # Focus layer配置
        self.focus_config = nnea_config.get('focus_layer', {})
        
        # Assist layer配置
        self.assist_config = nnea_config.get('assist_layer', {})
        self.use_assist_in_init = self.assist_config.get('use_in_init', True)
        self.assist_dropout = self.assist_config.get('dropout', 0.1)
        
        # TrainableGeneSetLayer参数 - 从nnea.geneset_layer获取
        geneset_config = nnea_config.get('geneset_layer', {})
        self.min_set_size = geneset_config.get('min_set_size', 10)
        self.max_set_size = geneset_config.get('max_set_size', 50)
        self.attention_dim = geneset_config.get('attention_dim', 32)
        self.geneset_dropout = geneset_config.get('dropout', 0.3)
        self.num_fc_layers = geneset_config.get('num_fc_layers', 0)
        self.geneset_threshold = geneset_config.get('geneset_threshold', 1e-5)
        
        # 构建网络层
        self._build_layers()
        
    def _is_classification_task(self) -> bool:
        """
        判断当前任务是否为分类任务
        
        Returns:
            是否为分类任务
        """
        return self.task_type == 'classification'
        
    def _build_layers(self):
        """构建网络层"""
        # 核心：TrainableGeneSetLayer
        self.geneset_layer = TrainableGeneSetLayer(
            num_genes=self.input_dim,
            num_sets=self.num_genesets,
            min_set_size=self.min_set_size,
            max_set_size=self.max_set_size,
            piror_knowledge=self.piror_knowledge,
            freeze_piror=self.freeze_piror,
            geneset_dropout=self.geneset_dropout,
            attention_dim=self.attention_dim,
            geneset_threshold=self.geneset_threshold
        )

        # 构建辅助层（assist_layer）- 用于初始化时直接映射概率
        self.assist_layer = nn.Sequential(
            nn.Linear(self.num_genesets, self.output_dim),
            nn.Dropout(self.assist_dropout),
        )
        
        # 构建focus_layer（集成hidden_layer、attention_layer和output_layer）
        self.focus_layer = self._build_focus_layer()
        
        # 初始化标志，用于控制是否使用assist_layer
        self.use_assist_layer = True
        
    def _build_focus_layer(self):
        """
        构建focus_layer，集成hidden_layer、attention_layer和output_layer
        参考train_utils中的BuildClassifier函数
        """
        # 获取配置参数
        focus_config = getattr(self, 'focus_config', {})
        classifier_name = focus_config.get('classifier_name', 'linear')
        hidden_dims = focus_config.get('hidden_dims', [])
        dropout_rates = focus_config.get('classifier_dropout', [self.dropout_rate] * len(hidden_dims))
        
        layers = []
        current_dim = self.num_genesets  # 基因集层的输出维度
        
        if classifier_name == "linear":
            # 线性分类器结构
            if len(hidden_dims) > 0:
                for i, h_dim in enumerate(hidden_dims):
                    layers.append(nn.Linear(current_dim, h_dim))
                    layers.append(nn.ReLU())
                    if i != len(hidden_dims) - 1:
                        layers.append(nn.BatchNorm1d(h_dim))
                        layers.append(nn.Dropout(dropout_rates[i] if i < len(dropout_rates) else self.dropout_rate))
                    current_dim = h_dim
            else:
                # 输出层
                layers.append(nn.ReLU())
            layers.append(nn.Linear(current_dim, self.output_dim))
            
        elif classifier_name == "attention":
            # 注意力分类器结构
            # 1. 添加自注意力层
            layers.append(AttentionBlock(current_dim))
            
            # 2. 添加MLP层（与linear分类器结构一致）
            if len(hidden_dims) > 0:
                for i, h_dim in enumerate(hidden_dims):
                    layers.append(nn.Linear(current_dim, h_dim))
                    layers.append(nn.ReLU())
                    if i != len(hidden_dims) - 1:
                        layers.append(nn.BatchNorm1d(h_dim))
                        layers.append(nn.Dropout(dropout_rates[i] if i < len(dropout_rates) else self.dropout_rate))
                    current_dim = h_dim
            
            # 3. 输出层
            layers.append(nn.Linear(current_dim, self.output_dim))
            
        else:
            # 默认结构（保持原有逻辑）
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU() if self.activation == 'relu' else nn.Tanh(),
                    nn.Dropout(self.dropout_rate)
                ])
                current_dim = hidden_dim
            
            # 注意力层
            layers.append(AttentionBlock(current_dim))
            
            # 输出层
            layers.append(nn.Linear(current_dim, self.output_dim))
        
        return nn.Sequential(*layers)
        
    def _prepare_input_for_geneset(self, x):
        """
        为基因集层准备输入数据
        
        Args:
            x: 输入张量 (batch_size, num_genes)
            
        Returns:
            R, S: 秩序矩阵和排列索引
        """
        # 计算每个样本的基因表达值排名（升序）
        rank_exp = torch.argsort(torch.argsort(x, dim=1, descending=False), dim=1, descending=False).float()
        
        # 生成降序排列索引
        sort_exp = torch.argsort(x, dim=1, descending=True)
        
        return rank_exp, sort_exp
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, num_genes)
            
        Returns:
            输出张量
        """
        # 为基因集层准备输入
        R, S = self._prepare_input_for_geneset(x)
        
        # 核心：TrainableGeneSetLayer
        geneset_output = self.geneset_layer(R, S)
        
        # 根据初始化标志选择使用assist_layer还是focus_layer
        if self.use_assist_layer:
            # 使用辅助层直接映射概率
            x = self.assist_layer(geneset_output)
            # 根据任务类型决定是否输出概率
            if self._is_classification_task():
                # 对于分类任务，应用log_softmax
                x = F.log_softmax(x, dim=1)
                x = torch.log(x + 1e-8)  # 添加小值避免log(0)
            # 对于回归、生存分析等任务，直接返回原始输出
        else:
            # 使用focus_layer（集成hidden_layer、attention_layer和output_layer）
            x = self.focus_layer(geneset_output)
            # 根据任务类型决定是否输出概率
            if self._is_classification_task():
                # 对于分类任务，应用log_softmax
                x = F.log_softmax(x, dim=1)
            # 对于回归、生存分析等任务，直接返回原始输出
        
        return x
    
    def get_geneset_importance(self) -> torch.Tensor:
        """
        获取基因集重要性
        
        Returns:
            基因集重要性张量
        """
        # 使用基因集指示矩阵的均值作为重要性
        indicators = self.geneset_layer.get_set_indicators()
        return torch.mean(indicators, dim=1)
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        获取注意力权重
        
        Returns:
            注意力权重张量
        """
        # 尝试从focus_layer中获取注意力权重
        try:
            # 查找focus_layer中的AttentionBlock
            for module in self.focus_layer.modules():
                if isinstance(module, AttentionBlock):
                    return module.get_attention_weights()
        except:
            pass
        
        # 如果没有找到注意力层，返回基因集重要性作为替代
        indicators = self.geneset_layer.get_set_indicators()
        return torch.mean(indicators, dim=1)
    
    def get_geneset_assignments(self) -> torch.Tensor:
        """
        获取基因到基因集的分配
        
        Returns:
            基因分配矩阵
        """
        return self.geneset_layer.get_set_indicators()
    
    def regularization_loss(self) -> torch.Tensor:
        """
        计算正则化损失
        
        Returns:
            正则化损失
        """
        reg_loss = self.geneset_layer.regularization_loss()

        return reg_loss
    
    def set_assist_layer_mode(self, use_assist: bool) -> None:
        """
        设置是否使用辅助层
        
        Args:
            use_assist: 是否使用辅助层
        """
        self.use_assist_layer = use_assist
        import logging
        logger = logging.getLogger(__name__)
        if use_assist:
            logger.info("已启用辅助层模式，将直接映射geneset输出为概率")
        else:
            logger.info("已启用标准模式，将使用focus_layer进行预测")
    
    def get_assist_layer_mode(self) -> bool:
        """
        获取当前辅助层使用状态
        
        Returns:
            是否使用辅助层
        """
        return self.use_assist_layer
    
    def get_task_type(self) -> str:
        """
        获取当前任务类型
        
        Returns:
            任务类型字符串
        """
        return self.task_type
