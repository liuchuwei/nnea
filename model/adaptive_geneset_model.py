import numpy as np
import torch.nn as nn
from model.nnea_layers import TrainableGeneSetLayer, FeatureAttention
import torch.nn.functional as F

class AdaptiveGeneSetModel(nn.Module):
    """
    自适应基因集富集分数模型
    整个模型可端到端训练

    参数:
    - num_genes: 总基因数
    - num_classes: 分类类别数
    - num_sets: 基因集数量 (默认20)
    - set_min_size: 基因集最小大小 (默认10)
    - set_max_size: 基因集最大大小 (默认50)
    - hidden_dim: 分类器隐藏层维度 (默认128)
    """

    def __init__(self, num_genes, num_classes, num_sets=20,
                 set_min_size=10, set_max_size=50, hidden_dim=128):
        super().__init__()

        # 可训练基因集层
        self.gene_set_layer = TrainableGeneSetLayer(
            num_genes=num_genes,
            num_sets=num_sets,
            min_set_size=set_min_size,
            max_set_size=set_max_size
        )

        # 分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(num_sets, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        #
        # self.classifier = nn.Sequential(
        #     nn.Linear(num_sets, hidden_dim),
        #     nn.ReLU(),
        #     FeatureAttention(hidden_dim),  # 插入特征注意力模块[1,5,8](@ref)
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Dropout(0.4),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     FeatureAttention(hidden_dim // 2),  # 第二层注意力[3,8](@ref)
        #     nn.BatchNorm1d(hidden_dim // 2),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_dim // 2, num_classes)
        # )

        # self.classifier = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(num_sets, num_classes)
        # )

        # 基因集重要性预测器
        self.set_importance_net = nn.Sequential(
            nn.Linear(num_sets, 1),
            nn.Sigmoid()
        )

    def forward(self, R, NR, S):
        # 计算富集分数
        es_scores = self.gene_set_layer(R, NR, S)

        # 预测类别
        logits = self.classifier(es_scores)

        # 计算基因集重要性
        set_importance = self.set_importance_net(es_scores)

        return F.log_softmax(logits, dim=1), set_importance.squeeze()

    def regularization_loss(self):
        """返回基因集层的正则化损失"""
        return self.gene_set_layer.regularization_loss()

    def get_gene_sets(self, x=None, threshold=0.5):
        """
        获取当前基因集的组成（基因列表）
        参数:
        - x: 输入数据 (可选，用于动态生成基因集)
        - threshold: 成员关系阈值

        返回:
        - 基因集字典: {set_idx: [gene_idx1, gene_idx2, ...]}
        """
        indicators = self.gene_set_layer.get_set_indicators(x)
        indicators = indicators.detach().cpu().numpy()

        gene_sets = {}
        for set_idx in range(indicators.shape[0]):
            # 选择超过阈值的基因
            member_mask = indicators[set_idx] > threshold
            gene_indices = np.where(member_mask)[0].tolist()
            gene_sets[f"Set_{set_idx}"] = gene_indices

        return gene_sets

