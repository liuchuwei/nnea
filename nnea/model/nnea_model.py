"""
NNEA (Neural Network with Explainable Architecture) Model
Implements explainable neural network architecture with support for gene set learning and biological interpretation
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
# Fix import path issues
import sys
import os
# Add project root directory to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
if project_root not in sys.path:
    sys.path.append(project_root)

class SafeBatchNorm1d(nn.BatchNorm1d):
    """
    A safe version of BatchNorm1d that automatically switches to eval mode
    when batch size is too small for training.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, input):
        # If batch size is 1 and we're in training mode, switch to eval mode temporarily
        if self.training and input.size(0) == 1:
            self.eval()
            output = super().forward(input)
            self.train()
            return output
        return super().forward(input)


class AttentionBlock(nn.Module):
    """自注意力模块，用于动态加权特征"""

    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)  # 压缩通道降低计算量
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的残差权重
        self.last_attention_weights = None  # 存储最后一次的注意力权重

    def forward(self, x):
        # x shape: [Batch, Sequence, Features] → 需调整为特征维度
        if x.dim() == 2:  # 若输入为2D（无序列长度），增加序列维度
            x = x.unsqueeze(1)  # [B, 1, D]

        Q = self.query(x)  # [B, Seq, D//8]
        K = self.key(x)  # [B, Seq, D//8]
        V = self.value(x)  # [B, Seq, D]

        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2))  # [B, Seq, Seq]
        attn_weights = F.softmax(scores, dim=-1)  # 归一化权重

        # 存储注意力权重
        self.last_attention_weights = attn_weights

        # 加权聚合特征
        context = torch.bmm(attn_weights, V)  # [B, Seq, D]

        # 残差连接 + 特征压缩回原始维度
        output = self.gamma * context + x
        return output.squeeze(1)  # 移除序列维度 → [B, D]

    def get_attention_weights(self):
        """获取注意力权重"""
        if self.last_attention_weights is not None:
            return self.last_attention_weights
        else:
            return torch.zeros(1)  # 返回占位符


logger = logging.getLogger(__name__)

class NNEAModel(nn.Module):
    """
    NNEA Neural Network Model
    Core component is TrainableGeneSetLayer, includes attention mechanism and explainability components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NNEA model
        
        Args:
            config: Model configuration
        """
        super(NNEAModel, self).__init__()
        
        # Device configuration
        device_config = config.get('device', 'cpu')
        if device_config == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Model parameters
        self.input_dim = config.get('input_dim', 0)
        self.output_dim = config.get('output_dim', 1)
        self.dropout_rate = config.get('dropout', 0.3)
        self.activation = config.get('activation', 'relu')
        
        # Task type - used to determine if probability output is needed
        self.task_type = config.get('global').get('task', 'classification')
        
        # Get nnea configuration section
        nnea_config = config.get('nnea', {})
        
        # NNEA specific parameters - get from nnea configuration
        self.use_piror_knowledge = nnea_config.get('piror_knowledge', {}).get('use_piror_knowledge', False)
        self.piror_knowledge = nnea_config.get('piror_knowledge', {}).get('piror_knowledge', None)
        self.freeze_piror = nnea_config.get('piror_knowledge', {}).get('freeze_piror', True)
        # If piror_knowledge exists, num_genesets equals the number of rows in piror_knowledge, otherwise use configuration
        if self.piror_knowledge is not None:
            self.num_genesets = self.piror_knowledge.shape[0]
        else:
            self.num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
        
        # Focus layer configuration
        self.focus_config = nnea_config.get('focus_layer', {})
        
        # Assist layer configuration
        self.assist_config = nnea_config.get('assist_layer', {})
        self.use_assist_in_init = self.assist_config.get('use_in_init', True)
        self.assist_dropout = self.assist_config.get('dropout', 0.1)
        self.assist_output = self.assist_config.get('output', 2)
        self.assist_type = self.assist_config.get('type', "rec")

        # TrainableGeneSetLayer parameters - get from nnea.geneset_layer
        geneset_config = nnea_config.get('geneset_layer', {})
        self.min_set_size = geneset_config.get('min_set_size', 10)
        self.max_set_size = geneset_config.get('max_set_size', 50)
        self.attention_dim = geneset_config.get('attention_dim', 32)
        self.geneset_dropout = geneset_config.get('dropout', 0.3)
        self.num_fc_layers = geneset_config.get('num_fc_layers', 0)
        self.geneset_threshold = geneset_config.get('geneset_threshold', 1e-5)
        
        # Build network layers
        self._build_layers()
        
    def _is_classification_task(self) -> bool:
        """
        Determine if current task is classification task
        
        Returns:
            Whether it is a classification task
        """
        return self.task_type == 'classification'
        
    def _build_layers(self):
        """Build network layers"""
        # Core: TrainableGeneSetLayer
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

        # Build assist layer - used for direct probability mapping during initialization
        self.assist_layer = nn.Sequential(
            nn.Linear(self.num_genesets, self.assist_output),
            nn.Dropout(self.assist_dropout),
        )
        
        # Build focus_layer (integrates hidden_layer, attention_layer and output_layer)
        self.focus_layer = self._build_focus_layer()
        
        # Initialization flag, used to control whether to use assist_layer
        self.use_assist_layer = True
        
    def _build_focus_layer(self):
        """
        Build focus_layer, integrating hidden_layer, attention_layer and output_layer
        Reference the BuildClassifier function in train_utils
        """
        # Get configuration parameters
        focus_config = getattr(self, 'focus_config', {})
        classifier_name = focus_config.get('classifier_name', 'linear')
        hidden_dims = focus_config.get('hidden_dims', [])
        dropout_rates = focus_config.get('classifier_dropout', [self.dropout_rate] * len(hidden_dims))
        
        layers = []
        current_dim = self.num_genesets  # Output dimension of gene set layer
        
        if classifier_name == "linear":
            # Linear classifier structure
            if len(hidden_dims) > 0:
                for i, h_dim in enumerate(hidden_dims):
                    layers.append(nn.Linear(current_dim, h_dim))
                    layers.append(nn.ReLU())
                    if i != len(hidden_dims) - 1:
                        layers.append(SafeBatchNorm1d(h_dim))
                        layers.append(nn.Dropout(dropout_rates[i] if i < len(dropout_rates) else self.dropout_rate))
                    current_dim = h_dim
            else:
                # Output layer
                layers.append(nn.ReLU())
            layers.append(nn.Linear(current_dim, self.output_dim))
            
        elif classifier_name == "attention":
            # Attention classifier structure
            # 1. Add self-attention layer
            layers.append(AttentionBlock(current_dim))
            
            # 2. Add MLP layers (same structure as linear classifier)
            if len(hidden_dims) > 0:
                for i, h_dim in enumerate(hidden_dims):
                    layers.append(nn.Linear(current_dim, h_dim))
                    layers.append(nn.ReLU())
                    if i != len(hidden_dims) - 1:
                        layers.append(SafeBatchNorm1d(h_dim))
                        layers.append(nn.Dropout(dropout_rates[i] if i < len(dropout_rates) else self.dropout_rate))
                    current_dim = h_dim
            
            # 3. Output layer
            layers.append(nn.Linear(current_dim, self.output_dim))
            
        else:
            # Default structure (maintain original logic)
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    SafeBatchNorm1d(hidden_dim),
                    nn.ReLU() if self.activation == 'relu' else nn.Tanh(),
                    nn.Dropout(self.dropout_rate)
                ])
                current_dim = hidden_dim
            
            # Attention layer
            layers.append(AttentionBlock(current_dim))
            
            # Output layer
            layers.append(nn.Linear(current_dim, self.output_dim))
        
        return nn.Sequential(*layers)
        
    def _prepare_input_for_geneset(self, x):
        """
        Prepare input data for gene set layer
        
        Args:
            x: Input tensor (batch_size, num_genes)
            
        Returns:
            R, S: Rank matrix and sort indices
        """
        # Calculate gene expression value ranking for each sample (ascending order)
        rank_exp = torch.argsort(torch.argsort(x, dim=1, descending=False), dim=1, descending=False).float()
        
        # Generate descending order sort indices
        sort_exp = torch.argsort(x, dim=1, descending=True)
        
        return rank_exp, sort_exp
        
    def forward(self, x):
        """
        Forward propagation
        
        Args:
            x: Input tensor (batch_size, num_genes)
            
        Returns:
            Output tensor
        """
        # Prepare input for gene set layer
        R, S = self._prepare_input_for_geneset(x)
        
        # Core: TrainableGeneSetLayer
        geneset_output = self.geneset_layer(R, S)
        
        # Choose between assist_layer and focus_layer based on initialization flag
        if self.use_assist_layer:
            # Use assist layer for direct probability mapping
            x = self.assist_layer(geneset_output)
            # Decide whether to output probability based on task type
            if self.assist_type == "classification":
                # For classification tasks, apply log_softmax
                x = F.log_softmax(x, dim=1)
            # For regression, survival analysis and other tasks, return original output directly
        else:
            # Use focus_layer (integrates hidden_layer, attention_layer and output_layer)
            x = self.focus_layer(geneset_output)
            # Decide whether to output probability based on task type
            if self._is_classification_task():
                # For classification tasks, apply log_softmax
                x = F.log_softmax(x, dim=1)
            # For regression, survival analysis and other tasks, return original output directly
        
        return x
    
    def get_geneset_importance(self) -> torch.Tensor:
        """
        Get gene set importance
        
        Returns:
            Gene set importance tensor
        """
        # Use mean of gene set indicator matrix as importance
        indicators = self.geneset_layer.get_set_indicators()
        return torch.mean(indicators, dim=1)
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        Get attention weights
        
        Returns:
            Attention weights tensor
        """
        # Try to get attention weights from focus_layer
        try:
            # Find AttentionBlock in focus_layer
            for module in self.focus_layer.modules():
                if isinstance(module, AttentionBlock):
                    return module.get_attention_weights()
        except:
            pass
        
        # If attention layer not found, return gene set importance as alternative
        indicators = self.geneset_layer.get_set_indicators()
        return torch.mean(indicators, dim=1)
    
    def get_geneset_assignments(self) -> torch.Tensor:
        """
        Get gene to gene set assignments
        
        Returns:
            Gene assignment matrix
        """
        return self.geneset_layer.get_set_indicators()
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Calculate regularization loss
        
        Returns:
            Regularization loss
        """
        reg_loss = self.geneset_layer.regularization_loss()

        return reg_loss
    
    def set_assist_layer_mode(self, use_assist: bool) -> None:
        """
        Set whether to use assist layer
        
        Args:
            use_assist: Whether to use assist layer
        """
        self.use_assist_layer = use_assist
        import logging
        logger = logging.getLogger(__name__)
        if use_assist:
            logger.info("Assist layer mode enabled, will directly map geneset output to probability")
        else:
            logger.info("Standard mode enabled, will use focus_layer for prediction")
    
    def get_assist_layer_mode(self) -> bool:
        """
        Get current assist layer usage status
        
        Returns:
            Whether assist layer is being used
        """
        return self.use_assist_layer
    
    def get_task_type(self) -> str:
        """
        Get current task type
        
        Returns:
            Task type string
        """
        return self.task_type
