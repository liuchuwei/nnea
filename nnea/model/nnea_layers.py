"""
NNEA专用神经网络层
包含基因集层、注意力层等可解释性组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any

class GeneSetLayer(nn.Module):
    """
    基因集层
    学习基因到基因集的映射关系，提供生物学可解释性
    """
    
    def __init__(self, input_dim: int, num_genesets: int = 100, mode: str = 'one_mode'):
        """
        初始化基因集层
        
        Args:
            input_dim: 输入维度（基因数量）
            num_genesets: 基因集数量
            mode: 模式 ('one_mode', 'multi_mode')
        """
        super(GeneSetLayer, self).__init__()
        
        self.input_dim = input_dim
        self.num_genesets = num_genesets
        self.mode = mode
        
        # 基因到基因集的映射矩阵（可学习的稀疏矩阵）
        self.gene_to_geneset = nn.Parameter(torch.randn(input_dim, num_genesets))
        
        # 稀疏性约束
        self.sparsity_weight = 0.1
        
        # 初始化稀疏性
        self._initialize_sparsity()
        
    def _initialize_sparsity(self):
        """初始化稀疏性"""
        # 使用稀疏初始化
        nn.init.sparse_(self.gene_to_geneset, sparsity=0.9)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            基因集表示 (batch_size, num_genesets)
        """
        # 应用稀疏性约束
        sparse_weights = F.softplus(self.gene_to_geneset)
        
        # 基因集聚合
        if self.mode == 'one_mode':
            # 每个基因只属于一个基因集
            geneset_assignments = F.softmax(self.gene_to_geneset, dim=1)
            geneset_output = torch.matmul(x, geneset_assignments)
        else:
            # 多模式：每个基因可以属于多个基因集
            geneset_weights = torch.sigmoid(self.gene_to_geneset)
            geneset_output = torch.matmul(x, geneset_weights)
        
        return geneset_output
    
    def get_importance(self) -> torch.Tensor:
        """
        获取基因集重要性
        
        Returns:
            基因集重要性张量
        """
        return torch.norm(self.gene_to_geneset, dim=0).detach()
    
    def get_gene_assignments(self) -> torch.Tensor:
        """
        获取基因到基因集的分配
        
        Returns:
            基因分配矩阵
        """
        if self.mode == 'one_mode':
            return F.softmax(self.gene_to_geneset, dim=1)
        else:
            return torch.sigmoid(self.gene_to_geneset)
    
    def sparsity_loss(self) -> torch.Tensor:
        """
        计算稀疏性损失
        
        Returns:
            稀疏性损失
        """
        return self.sparsity_weight * torch.norm(self.gene_to_geneset, p=1)

class TrainableGeneSetLayer(nn.Module):
    """
    可训练基因集层 - 基因集的成员关系作为可训练参数
    包含注意力机制用于动态调整基因重要性

    参数:
    - num_genes: 总基因数
    - num_sets: 基因集数量
    - min_set_size: 基因集最小基因数（通过稀疏性约束实现）
    - max_set_size: 基因集最大基因数
    - alpha: 富集分数计算中的指数参数
    - attention_dim: 注意力机制隐藏层维度
    """

    def __init__(self, num_genes, num_sets, min_set_size=10, max_set_size=50,
                 alpha=0.25, num_fc_layers=0, is_deep_layer=False, layer_index=0,
                 prior_knowledge=None, freeze_prior=True, geneset_dropout=0.3,
                 use_attention=False, attention_dim=64, geneset_threshold=0.0):
        super().__init__()
        self.num_genes = num_genes
        self.num_sets = num_sets
        self.alpha = nn.Parameter(torch.tensor(-1.0))
        self.is_deep_layer = is_deep_layer
        self.layer_index = layer_index
        self.prior_knowledge = prior_knowledge
        self.num_fc_layers = num_fc_layers
        self.geneset_dropout = nn.Dropout(p=geneset_dropout)
        self.use_attention = use_attention
        self.attention_dim = attention_dim
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.geneset_threshold = 1e-5

        # 新增可配置参数
        self.size_reg_weight = nn.Parameter(torch.tensor(0.1))
        self.diversity_reg_weight = nn.Parameter(torch.tensor(0.5))

        # 初始化成员关系矩阵
        if not use_attention:
            if prior_knowledge is not None:
                assert prior_knowledge.shape[1] == num_genes
                # 将0/1矩阵转换为大数值参数（sigmoid后接近0或1）
                adjusted_prior = torch.zeros_like(prior_knowledge)
                adjusted_prior[prior_knowledge > 0.5] = 3  # sigmoid(10) ≈ 1
                adjusted_prior[prior_knowledge <= 0.5] = -3  # sigmoid(-10) ≈ 0

                # 存储原始先验知识矩阵（不参与训练）
                self.register_buffer("prior_mask", prior_knowledge.float())
                # 可训练参数（允许微调时更新）
                self.set_membership = nn.Parameter(adjusted_prior, requires_grad=not freeze_prior)
            else:
                self.set_membership = nn.Parameter(torch.randn(num_sets, num_genes))
                self.prior_mask = None

        else:
            # 注意力模式：使用基因嵌入和查询向量
            self.gene_embeddings = nn.Parameter(torch.randn(num_genes, attention_dim))
            self.query_vectors = nn.Parameter(torch.randn(num_sets, attention_dim))

            # 处理先验知识
            if prior_knowledge is not None:
                self.register_buffer("prior_mask", prior_knowledge.float())
            else:
                self.prior_mask = None

        # 设置稀疏性约束以控制基因集大小
        self.min_set_size = min_set_size
        self.max_set_size = max_set_size

        # 构建全连接变换层（当num_fc_layers>0时）
        if num_fc_layers > 0:
            # 共享权重设计：所有基因集使用相同的变换层
            self.fc_transform = self._build_fc_transform(num_genes, num_fc_layers)
        else:
            self.fc_transform = None

    def _build_fc_transform(self, num_genes, num_layers):
        """构建全连接变换模块"""
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(num_genes, num_genes))
            # 最后一层不加激活函数（保持线性变换）
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def get_set_indicators(self, x=None):
        """
        获取基因集指示矩阵，在训练过程中动态调整
        返回软指示矩阵（可微）

        参数:
        - x: 输入基因表达数据 (batch_size, num_genes)，用于注意力计算

        返回:
        - indicators: 基因集指示矩阵 (num_sets, num_genes)
        """
        if self.use_attention:
            # 计算注意力分数：查询向量与基因嵌入的点积
            attn_scores = torch.matmul(self.query_vectors, self.gene_embeddings.t())

            # 应用温度缩放控制稀疏性
            scaled_scores = attn_scores / torch.clamp(self.temperature, min=0.01)

            # 使用sigmoid生成指示矩阵（保持稀疏性）
            indicators = torch.sigmoid(scaled_scores)

            # 应用先验知识约束
            if self.prior_mask is not None:
                indicators = indicators * self.prior_mask

            # 应用dropout
            indicators = self.geneset_dropout(indicators)

        else:
            if self.fc_transform is not None and not (hasattr(self, 'freeze_prior') and self.freeze_prior and self.prior_knowledge is not None):
                # 保留原始值直通（残差连接）以便必要时恢复原始行为
                transformed = self.set_membership + 0.3*self.fc_transform(self.set_membership)
            else:
                transformed = self.set_membership

            # 基础成员关系矩阵
            indicators = torch.sigmoid(transformed)

            if self.prior_mask is not None:
                # 将先验知识作为硬性约束：非先验位置强制归零
                indicators = indicators * self.prior_mask
            else:
                indicators = indicators

            # 应用稀疏性约束
            # 确保每个基因集达到最小大小
            if not (self.prior_mask is not None and hasattr(self, 'freeze_prior') and self.freeze_prior):
                avg_indicators = indicators.mean(dim=1, keepdim=True)
                indicators = torch.where(
                    indicators < avg_indicators * 0.3,
                    indicators * 0.01,
                    indicators
                )

            indicators = self.geneset_dropout(indicators)
        return indicators

    def regularization_loss(self):
        """
        基因集的正则化损失，确保基因集在期望大小范围内
        """
        # 获取基因指示矩阵
        indicators = self.get_set_indicators()

        if self.prior_knowledge is None or not (hasattr(self, 'freeze_prior') and self.freeze_prior):
            # 计算每个基因集的"基因数"（期望值）
            set_sizes = torch.sum(indicators, dim=1)

            # 约束1: 防止基因集过小
            size_min_loss = F.relu(self.min_set_size - set_sizes).mean()

            # 约束2: 防止基因集过大
            size_max_loss = F.relu(set_sizes - self.max_set_size).mean()

        # 约束3: 鼓励清晰的成员关系（二值化）
        safe_log = torch.log(torch.clamp(indicators, min=1e-8))
        entropy_loss = (-indicators * safe_log).mean()

        # 约束4: 基因集多样性（防止重叠）
        if indicators.size()[0] > 1:
            normalized_indicators = indicators / (indicators.norm(dim=1, keepdim=True) + 1e-8)
            overlap_matrix = torch.mm(normalized_indicators, normalized_indicators.t())
            diversity_loss = torch.triu(overlap_matrix, diagonal=1).sum() / (self.num_sets * (self.num_sets - 1) // 2)
        else:
            diversity_loss = 0

        if self.is_deep_layer:
            exponent = torch.tensor(-self.layer_index * 0.5, device=size_min_loss.device)
            depth_loss = torch.exp(exponent) * (size_min_loss + size_max_loss)
            base_loss = 0.05 * depth_loss + 0.2 * entropy_loss + 0.5 * diversity_loss
        else:
            base_loss = 0.1 * size_min_loss + 0.1 * size_max_loss + 0.2 * entropy_loss + 0.5 * diversity_loss

        return base_loss

    def forward(self, R, S, mask=None):
        """
        输入:
        - R: 基因表达数据 (batch_size, num_genes)
        - S: 排序索引 (batch_size, num_genes)
        - mask: 掩码 (可选)

        输出:
        - es_scores: 富集分数 (batch_size, num_sets)
        """
        batch_size, num_genes = R.shape
        device = R.device

        # 获取基因集指示矩阵 (num_sets, num_genes)
        indicators = self.get_set_indicators().to(device)

        # 如果提供了mask，将其应用到指示矩阵上
        if mask is not None:
            indicators = indicators * mask.unsqueeze(0)

        sorted_indicators = torch.gather(
            indicators.unsqueeze(0).expand(batch_size, -1, -1),
            dim=2,
            index=S.long().unsqueeze(1).expand(-1, indicators.size(0), -1)
        )

        R_sorted = torch.gather(R.unsqueeze(1).expand(-1, indicators.size(0), -1),
                                dim=2,
                                index=S.long().unsqueeze(1).expand(-1, indicators.size(0), -1))

        # 计算正集加权
        clamped_input = torch.clamp(R_sorted * sorted_indicators, min=1e-8, max=1e4)
        weighted_pos = clamped_input ** torch.sigmoid(self.alpha)
        sum_weighted_pos = torch.sum(weighted_pos, dim=2, keepdim=True)

        # 计算正集累积分布
        cumsum_pos = torch.cumsum(weighted_pos, dim=2)
        valid_pos = sum_weighted_pos > 1e-8
        step_cdf_pos = torch.where(
            valid_pos,
            cumsum_pos / (sum_weighted_pos + 1e-10),
            torch.zeros_like(cumsum_pos)
        )

        # 计算负集累积分布
        neg_indicators = sorted_indicators < self.geneset_threshold
        if mask is not None:
            batch_size = R.shape[0]
            mask_sorted = torch.gather(
                mask.unsqueeze(0).expand(batch_size, -1),
                dim=1,
                index=S.long()
            )
            mask_expanded = mask_sorted.unsqueeze(1).expand(-1, self.num_sets, -1)
            neg_indicators = neg_indicators * mask_expanded

        sum_neg = torch.sum(neg_indicators, dim=2, keepdim=True)
        cumsum_neg = torch.cumsum(neg_indicators, dim=2)
        valid_neg = sum_neg > 1e-8
        step_cdf_neg = torch.where(
            valid_neg,
            cumsum_neg / (sum_neg + 1e-10),
            torch.zeros_like(cumsum_neg)
        )

        # 计算富集分数
        diff = (step_cdf_pos - step_cdf_neg) / num_genes
        es_scores = torch.sum(diff, dim=2)

        return es_scores

class AttentionLayer(nn.Module):
    """
    注意力层
    学习特征的重要性权重，提供可解释性
    """
    
    def __init__(self, input_dim: int, num_heads: int = 4):
        """
        初始化注意力层
        
        Args:
            input_dim: 输入维度
            num_heads: 注意力头数
        """
        super(AttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # 确保head_dim是整数
        if self.head_dim == 0:
            self.head_dim = 1
            self.num_heads = input_dim
        
        # 注意力权重
        self.attention_weights = nn.Parameter(torch.randn(input_dim))
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=min(num_heads, input_dim),
            batch_first=True
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            注意力加权输出 (batch_size, input_dim)
        """
        # 计算注意力权重
        attention_scores = torch.sigmoid(self.attention_weights)
        
        # 应用注意力
        attended_x = x * attention_scores.unsqueeze(0)
        
        # 多头注意力（如果需要）
        if self.num_heads > 1:
            # 重塑为序列形式
            x_seq = attended_x.unsqueeze(1)  # (batch_size, 1, input_dim)
            attn_output, _ = self.multihead_attn(x_seq, x_seq, x_seq)
            attended_x = attn_output.squeeze(1)
        
        return attended_x
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        获取注意力权重
        
        Returns:
            注意力权重张量
        """
        return torch.sigmoid(self.attention_weights).detach()

class ExplainableLayer(nn.Module):
    """
    可解释层
    提供模型解释的中间表示
    """
    
    def __init__(self, input_dim: int, explanation_dim: int = 64):
        """
        初始化可解释层
        
        Args:
            input_dim: 输入维度
            explanation_dim: 解释维度
        """
        super(ExplainableLayer, self).__init__()
        
        self.input_dim = input_dim
        self.explanation_dim = explanation_dim
        
        # 解释投影层
        self.explanation_projection = nn.Linear(input_dim, explanation_dim)
        
        # 重要性权重
        self.importance_weights = nn.Parameter(torch.ones(input_dim))
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            可解释表示 (batch_size, explanation_dim)
        """
        # 应用重要性权重
        weighted_x = x * torch.sigmoid(self.importance_weights).unsqueeze(0)
        
        # 投影到解释空间
        explanation = self.explanation_projection(weighted_x)
        
        return explanation
    
    def get_feature_importance(self) -> torch.Tensor:
        """
        获取特征重要性
        
        Returns:
            特征重要性张量
        """
        return torch.sigmoid(self.importance_weights)
    
    def get_explanation_components(self) -> torch.Tensor:
        """
        获取解释组件
        
        Returns:
            解释组件张量
        """
        return self.explanation_projection.weight

class BiologicalConstraintLayer(nn.Module):
    """
    生物学约束层
    应用生物学先验知识约束
    """
    
    def __init__(self, input_dim: int, prior_knowledge: Optional[torch.Tensor] = None):
        """
        初始化生物学约束层
        
        Args:
            input_dim: 输入维度
            prior_knowledge: 先验知识矩阵
        """
        super(BiologicalConstraintLayer, self).__init__()
        
        self.input_dim = input_dim
        self.prior_knowledge = prior_knowledge
        
        # 约束权重
        self.constraint_weight = 0.1
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            约束后的输出
        """
        # 如果有先验知识，应用约束
        if self.prior_knowledge is not None:
            # 应用先验知识约束
            constrained_x = x * self.prior_knowledge.unsqueeze(0)
            return constrained_x
        
        return x
    
    def constraint_loss(self) -> torch.Tensor:
        """
        计算约束损失
        
        Returns:
            约束损失
        """
        if self.prior_knowledge is not None:
            # 基于先验知识的约束损失
            return self.constraint_weight * torch.norm(self.prior_knowledge, p=1)
        return torch.tensor(0.0)
    
    def set_prior_knowledge(self, prior_knowledge: torch.Tensor):
        """
        设置先验知识
        
        Args:
            prior_knowledge: 先验知识矩阵
        """
        self.prior_knowledge = prior_knowledge 