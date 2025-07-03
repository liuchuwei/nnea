import torch
import torch.nn as nn
import torch.nn.functional as F



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
                 alpha=0.25, is_deep_layer=False, layer_index=0,
                 prior_knowledge=None, freeze_prior=True
                 ):
        super().__init__()
        self.num_genes = num_genes
        self.num_sets = num_sets
        self.alpha = alpha
        self.is_deep_layer = is_deep_layer
        self.layer_index = layer_index

        # 初始化成员关系矩阵
        if prior_knowledge is not None:
            # 确保先验矩阵与基因数一致
            assert prior_knowledge.shape[1] == num_genes, "Prior knowledge dimension mismatch"
            self.set_membership = nn.Parameter(prior_knowledge.float(), requires_grad=not freeze_prior)
        else:
            self.set_membership = nn.Parameter(torch.randn(num_sets, num_genes))


        # 设置稀疏性约束以控制基因集大小

        self.min_set_size = min_set_size
        self.max_set_size = max_set_size

    def get_set_indicators(self, x=None):
        """
        获取基因集指示矩阵，在训练过程中动态调整
        返回软指示矩阵（可微）

        参数:
        - x: 输入基因表达数据 (batch_size, num_genes)，用于注意力计算

        返回:
        - indicators: 基因集指示矩阵 (num_sets, num_genes)
        """

        # 基础成员关系矩阵
        indicators = torch.sigmoid(self.set_membership)
        # 如果提供输入数据，应用注意力机制

        # 应用稀疏性约束
        # 确保每个基因集达到最小大小
        avg_indicators = indicators.mean(dim=1, keepdim=True)
        indicators = torch.where(
            indicators < avg_indicators * 0.3,
            indicators * 0.01,
            indicators
        )

        return indicators

    def regularization_loss(self):
        """
        基因集的正则化损失，确保基因集在期望大小范围内
        """

        # 获取基因指示矩阵
        indicators = self.get_set_indicators()

        if self.prior_knowledge is None or not self.freeze_prior:

            # 计算每个基因集的"基因数"（期望值）
            set_sizes = torch.sum(indicators, dim=1)

            # 约束1: 防止基因集过小
            size_min_loss = F.relu(self.min_set_size - set_sizes).mean()

            # 约束2: 防止基因集过大
            size_max_loss = F.relu(set_sizes - self.max_set_size).mean()

        # 约束3: 鼓励清晰的成员关系（二值化）
        # entropy_loss = (-indicators * torch.log(indicators + 1e-8)).mean()
        safe_log = torch.log(torch.clamp(indicators, min=1e-8))
        entropy_loss = (-indicators * safe_log).mean()

        # 约束4: 基因集多样性（防止重叠）
        normalized_indicators = indicators / (indicators.norm(dim=1, keepdim=True) + 1e-8)
        overlap_matrix = torch.mm(normalized_indicators, normalized_indicators.t())
        diversity_loss = torch.triu(overlap_matrix, diagonal=1).sum() / (self.num_sets * (self.num_sets - 1) // 2)

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
        - x: 基因表达数据 (batch_size, num_genes)

        输出:
        - es_scores: 富集分数 (batch_size, num_sets)
        """
        batch_size, num_genes = R.shape
        device = R.device

        # 获取基因集指示矩阵 (num_sets, num_genes)
        indicators = self.get_set_indicators().to(device)  # 确保与R同设备

        # 如果提供了mask，将其应用到指示矩阵上
        if mask is not None:
            indicators = indicators * mask.unsqueeze(
                0)  # (num_sets, num_genes) * (1, num_genes) -> (num_sets, num_genes)

        sorted_indicators = torch.gather(
            indicators.unsqueeze(0).expand(batch_size, -1, -1),
            dim=2,
            index=S.long().unsqueeze(1).expand(-1, indicators.size(0), -1)
        )  # (batch_size, num_sets, num_genes)

        R_sorted = torch.gather(R.unsqueeze(1).expand(-1, indicators.size(0), -1),
                                dim=2,
                                index=S.long().unsqueeze(1).expand(-1, indicators.size(0), -1))

        # 计算正集加权 (alpha=0.25)
        clamped_input = torch.clamp(R_sorted * sorted_indicators, min=1e-8, max=1e4)
        # weighted_pos = (R_sorted * sorted_indicators) ** self.alpha
        weighted_pos = clamped_input ** self.alpha
        sum_weighted_pos = torch.sum(weighted_pos, dim=2, keepdim=True)  # (batch_size, num_sets, 1)

        # 计算正集累积分布 (避免除零)
        cumsum_pos = torch.cumsum(weighted_pos, dim=2)
        valid_pos = sum_weighted_pos > 1e-8
        step_cdf_pos = torch.where(
            valid_pos,  # 条件掩码（自动广播）
            # cumsum_pos / sum_weighted_pos,  # 真值计算
            cumsum_pos / (sum_weighted_pos + 1e-10),  # 真值计算
            torch.zeros_like(cumsum_pos)  # 假值填充
        )
        # 计算负集累积分布
        neg_indicators = sorted_indicators<0.1
        if mask is not None:
            # 高效生成排序后的基因掩码
            batch_size = R.shape[0]
            # 扩展mask并应用排序索引
            mask_sorted = torch.gather(
                mask.unsqueeze(0).expand(batch_size, -1),  # 扩展为(batch_size, num_genes)
                dim=1,
                index=S.long()  # 直接使用原始排序索引
            )
            # 调整维度适配负集计算
            mask_expanded = mask_sorted.unsqueeze(1).expand(-1, self.num_sets, -1)

            # 应用到负集指示矩阵
            neg_indicators = neg_indicators * mask_expanded

        sum_neg = torch.sum(neg_indicators, dim=2, keepdim=True)
        cumsum_neg = torch.cumsum(neg_indicators, dim=2)
        # step_cdf_neg = torch.zeros_like(cumsum_neg)
        valid_neg = sum_neg > 1e-8
        step_cdf_neg = torch.where(
            valid_neg,  # 条件掩码（自动广播）
            # cumsum_neg / sum_neg,  # 真值计算
            cumsum_neg / (sum_neg + 1e-10),  # 真值计算
            torch.zeros_like(cumsum_neg)  # 假值填充
        )

        # 计算富集分数
        diff = (step_cdf_pos - step_cdf_neg) / num_genes
        es_scores = torch.sum(diff, dim=2)

        return es_scores


class AttentionClassifier(nn.Module):
    """
    基于注意力的可解释分类器
    使用注意力权重展示基因集对分类决策的贡献

    参数:
    - num_sets: 基因集数量
    - num_classes: 类别数量
    - hidden_dim: 注意力机制的隐藏维度
    """

    def __init__(self, num_sets, num_classes, hidden_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_sets = num_sets
        self.hidden_dim = hidden_dim

        # 初始化查询向量（每个类别一个）
        self.query_vectors = nn.Parameter(torch.randn(num_classes, hidden_dim))

        # 初始化键向量（每个基因集一个）
        self.key_vectors = nn.Parameter(torch.randn(num_sets, hidden_dim))

        # 类别偏置项
        self.bias = nn.Parameter(torch.zeros(num_classes))

        # 初始化参数
        nn.init.xavier_uniform_(self.query_vectors)
        nn.init.xavier_uniform_(self.key_vectors)

    def forward(self, es_scores):
        """
        输入:
        - es_scores: 富集分数，形状 (batch_size, num_sets)

        输出:
        - logits: 类别分数，形状 (batch_size, num_classes)
        - attention_weights: 注意力权重，形状 (batch_size, num_classes, num_sets)
        """
        batch_size = es_scores.size(0)

        # 1. 计算基础注意力分数
        # 计算查询向量和键向量的相似度
        base_attn = torch.matmul(self.query_vectors, self.key_vectors.t())  # (num_classes, num_sets)
        base_attn = base_attn / (self.hidden_dim ** 0.5)  # 缩放

        # 2. 加入样本特定的富集分数信息
        # 使用富集分数调整注意力分数
        attn_scores = base_attn.unsqueeze(0) * es_scores.unsqueeze(1)  # (batch_size, num_classes, num_sets)

        # 3. 计算注意力权重
        attention_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_classes, num_sets)

        # 4. 使用注意力权重聚合富集分数
        # 展示每个基因集对各类别的贡献
        logits = torch.matmul(attention_weights, es_scores.unsqueeze(-1)).squeeze(-1)  # (batch_size, num_classes)

        # 5. 添加偏置项
        logits = logits + self.bias.unsqueeze(0)

        return logits, attention_weights

# class TrainableGeneSetLayer(nn.Module):
#     """
#     可训练基因集层 - 基因集的成员关系作为可训练参数
#     包含注意力机制用于动态调整基因重要性
#
#     参数:
#     - num_genes: 总基因数
#     - num_sets: 基因集数量
#     - min_set_size: 基因集最小基因数（通过稀疏性约束实现）
#     - max_set_size: 基因集最大基因数
#     - alpha: 富集分数计算中的指数参数
#     - attention_dim: 注意力机制隐藏层维度
#     """
#
#     def __init__(self, num_genes, num_sets, min_set_size=10, max_set_size=50,
#                  alpha=0.25, attention_dim=16,
#                  mask_mode=False,
#                  atten_mode=False,
#                  gumbel_temperature=0.1,
#                  temperature=1.0,
#                  adaptive_num_sets=False,
#                  importance_l1_weight=0.01,
#                  thersold = None,
#                  ):
#         super().__init__()
#         self.num_genes = num_genes
#         self.num_sets = num_sets
#         self.alpha = alpha
#         self.temperature = temperature
#         self.atten_mode = atten_mode
#         self.adaptive_num_sets = adaptive_num_sets
#         self.importance_l1_weight = importance_l1_weight
#         self.gumbel_temperature = gumbel_temperature
#         self.mask_mode = mask_mode
#         self.thersold = thersold
#         # 可训练参数：基因集成员关系 (num_sets, num_genes)
#         # 初始化：均匀分布，后续通过稀疏性约束控制基因集大小
#         self.set_membership = nn.Parameter(torch.randn(num_sets, num_genes))
#         self.set_importance = nn.Parameter(torch.ones(num_sets))  # 初始所有基因集同等重要
#
#         # self.set_membership = self.sparse_row_init(self.set_membership, min_nonzero=min_set_size, max_nonzero=max_set_size, std=0.01)
#
#         # 基因集注意力机制
#         self.attention_net = nn.Sequential(
#             nn.Linear(1, attention_dim),
#             nn.ReLU(),
#             nn.Linear(attention_dim, attention_dim),
#             nn.ReLU(),
#             nn.Linear(attention_dim, num_sets)
#         )
#
#         self.gene_attention = GeneAttention(
#             num_genes=num_genes,
#             num_sets=num_sets,
#             key_dim=16,
#             num_heads=4
#         )
#         # 基因集全局重要性权重
#         self.set_weights = nn.Parameter(torch.randn(num_sets))
#
#         # 设置稀疏性约束以控制基因集大小
#         self.min_set_size = min_set_size
#         self.max_set_size = max_set_size
#
#         # 获取mask indicator
#         self.mask_mode = mask_mode
#         self.gumbel_temperature = gumbel_temperature
#         if mask_mode:
#             self.logits = nn.Parameter(torch.randn(num_genes, num_sets, 2))
#
#     def sparse_row_init(self, tensor, min_nonzero=10, max_nonzero=50, std=0.01):
#         """
#         稀疏初始化矩阵，确保每行有[min_nonzero, max_nonzero]个非零值
#
#         参数:
#         tensor: (num_sets, num_genes) 待初始化的矩阵
#         min_nonzero: 每行最小非零元素数
#         max_nonzero: 每行最大非零元素数
#         std: 非零元素的正态分布标准差
#         """
#         num_sets, num_genes = tensor.shape
#
#         with torch.no_grad():  # 禁用梯度计算
#             # 1. 初始化为全零矩阵
#             tensor.zero_()
#
#             # 2. 逐行初始化
#             for i in range(num_sets):
#                 # 随机确定当前行的非零元素数量
#                 num_nonzero = torch.randint(min_nonzero, max_nonzero + 1, (1,)).item()
#
#                 # 随机选择基因索引（不放回抽样）
#                 gene_indices = torch.randperm(num_genes)[:num_nonzero]
#
#                 # 生成随机值（正态分布）
#                 random_values = torch.randn(num_nonzero) * std
#
#                 # 将随机值赋给选定位置
#                 tensor[i, gene_indices] = random_values
#
#         return tensor
#
#     def get_set_indicators(self, x=None):
#         """
#         获取基因集指示矩阵，在训练过程中动态调整
#         返回软指示矩阵（可微）
#
#         参数:
#         - x: 输入基因表达数据 (batch_size, num_genes)，用于注意力计算
#
#         返回:
#         - indicators: 基因集指示矩阵 (num_sets, num_genes)
#         """
#
#         if self.mask_mode:
#             # 2. 动态计算Gumbel Softmax（确保正确设备）
#             # mask = F.gumbel_softmax(
#             #     self.logits,
#             #     tau=self.gumbel_temperature,
#             #     hard=True
#             # )
#             # indicators = mask[:, :, 1].permute(1, 0)  # (num_sets, num_genes)
#             indicators =  torch.sigmoid(self.logits)
#
#         elif self.atten_mode:
#             indicators = self.gene_attention(x.unsqueeze(-1))  # (batch, sets, genes)
#             indicators = indicators.mean(dim=0)
#
#             # # 应用稀疏性约束
#             # # 确保每个基因集达到最小大小
#             # avg_indicators = indicators.mean(dim=1, keepdim=True)
#             # indicators = torch.where(
#             #     indicators < avg_indicators * 0.3,
#             #     indicators * 0.01,
#             #     indicators
#             # )
#             self.indicators = indicators
#
#         else:
#             # 基础成员关系矩阵
#             indicators = torch.sigmoid(self.set_membership)
#             # indicators = self.set_membership
#             # 如果提供输入数据，应用注意力机制
#             if x is not None and x.ndim == 2:
#                 # 计算基因级别的注意力权重
#                 # attention_input = x.mean(dim=0).unsqueeze(-1)  # 使用平均表达值
#                 # gene_attention = torch.sigmoid(self.attention_net(x.unsqueeze(-1)))
#                 gene_attention = self.gene_attention(x.unsqueeze(-1))  # (batch, sets, genes)
#
#                 # 按基因集维度转置
#                 # gene_attention = gene_attention.permute(1, 0)
#
#                 # 应用注意力调整基因集成员关系
#                 indicators = indicators * gene_attention
#                 # indicators = indicators.unsqueeze(0) * gene_attention  # (batch, sets, genes)
#                 indicators = indicators.mean(dim=0)
#
#             # 应用稀疏性约束
#             # 确保每个基因集达到最小大小
#             avg_indicators = indicators.mean(dim=1, keepdim=True)
#             indicators = torch.where(
#                 indicators < avg_indicators * 0.3,
#                 indicators * 0.01,
#                 indicators
#             )
#             # 应用基因集重要性权重
#         if self.adaptive_num_sets:
#             importance_weights = torch.sigmoid(self.set_importance).view(-1, 1)
#             indicators = indicators * importance_weights
#
#         return indicators
#
#     def regularization_loss(self):
#         """
#         基因集的正则化损失，确保基因集在期望大小范围内
#         """
#         if self.mask_mode:
#             indicators = self.get_set_indicators()
#             indicators = indicators[:,:,0]
#         elif self.atten_mode:
#             indicators = self.indicators
#
#         else:
#             indicators = torch.sigmoid(self.set_membership)
#
#         # 计算每个基因集的"基因数"（期望值）
#         set_sizes = torch.sum(indicators, dim=1)
#
#         # 约束1: 防止基因集过小
#         size_min_loss = F.relu(self.min_set_size - set_sizes).mean()
#
#         # 约束2: 防止基因集过大
#         size_max_loss = F.relu(set_sizes - self.max_set_size).mean()
#
#         # 约束3: 鼓励清晰的成员关系（二值化）
#         entropy_loss = (-indicators * torch.log(indicators + 1e-8)).mean()
#
#         # 约束4: 基因集多样性（防止重叠）
#         normalized_indicators = indicators / (indicators.norm(dim=1, keepdim=True) + 1e-8)
#         overlap_matrix = torch.mm(normalized_indicators, normalized_indicators.t())
#         diversity_loss = torch.triu(overlap_matrix, diagonal=1).sum() / (self.num_sets * (self.num_sets - 1) // 2)
#
#         # 基因集重要性稀疏约束（L1正则化）
#         importance_loss = torch.norm(torch.sigmoid(self.set_importance), p=1) * self.importance_l1_weight
#
#         base_loss = 0.1 * size_min_loss + 0.1 * size_max_loss + 0.2 * entropy_loss + 0.5 * diversity_loss
#
#         return base_loss + importance_loss if self.adaptive_num_sets else base_loss
#
#     def soft_rank(self, values, descending=True):
#         """可微的软排序操作"""
#         # 计算两两差异
#         pairwise_diff = values.unsqueeze(1) - values.unsqueeze(0)
#
#         # 应用sigmoid函数实现软排序
#         if descending:
#             ranks = torch.sigmoid(pairwise_diff / self.temperature).sum(dim=1)
#         else:
#             ranks = torch.sigmoid(-pairwise_diff / self.temperature).sum(dim=1)
#
#         return ranks
#
#     def forward(self, R, S):
#         """
#         输入:
#         - x: 基因表达数据 (batch_size, num_genes)
#
#         输出:
#         - es_scores: 富集分数 (batch_size, num_sets)
#         """
#         batch_size, num_genes = R.shape
#         device = R.device
#
#         # 获取基因集指示矩阵 (num_sets, num_genes)
#         # indicators = self.get_set_indicators(x=NR).to(device)  # 确保与R同设备
#         indicators = self.get_set_indicators().to(device)  # 确保与R同设备
#
#         # 准备排序后的数据
#         if self.mask_mode:
#             pos_indicators = indicators[:,:,0]
#             neg_indicators = indicators[:,:,1]
#
#             pos_sorted_indicators = torch.gather(
#                 pos_indicators.unsqueeze(0).expand(batch_size, -1, -1),
#                 dim=2,
#                 index=S.unsqueeze(1).expand(-1, pos_indicators.size(0), -1)
#             )  # (batch_size, num_sets, num_genes)
#
#             pos_R_sorted = torch.gather(R.unsqueeze(1).expand(-1, pos_indicators.size(0), -1),
#                                     dim=2,
#                                     index=S.unsqueeze(1).expand(-1, pos_indicators.size(0), -1))
#
#             neg_sorted_indicators = torch.gather(
#                 neg_indicators.unsqueeze(0).expand(batch_size, -1, -1),
#                 dim=2,
#                 index=S.unsqueeze(1).expand(-1, neg_indicators.size(0), -1)
#             )  # (batch_size, num_sets, num_genes)
#
#             neg_R_sorted = torch.gather(R.unsqueeze(1).expand(-1, neg_indicators.size(0), -1),
#                                         dim=2,
#                                         index=S.unsqueeze(1).expand(-1, neg_indicators.size(0), -1))
#
#             # 计算正集加权 (alpha=0.25)
#             alpha = 0.25
#             weighted_pos = (pos_R_sorted * pos_sorted_indicators) ** alpha
#             sum_weighted_pos = torch.sum(weighted_pos, dim=2, keepdim=True)  # (batch_size, num_sets, 1)
#
#             # 计算正集累积分布 (避免除零)
#             cumsum_pos = torch.cumsum(weighted_pos, dim=2)
#             # step_cdf_pos = torch.zeros_like(cumsum_pos)
#             valid_pos = sum_weighted_pos > 1e-8
#             # step_cdf_pos[valid_pos] = cumsum_pos[valid_pos] / sum_weighted_pos[valid_pos]
#             step_cdf_pos = torch.where(
#                 valid_pos,  # 条件掩码（自动广播）
#                 cumsum_pos / sum_weighted_pos,  # 真值计算
#                 torch.zeros_like(cumsum_pos)  # 假值填充
#             )
#             # 计算负集累积分布
#             # neg_indicators = 1.0 - sorted_indicators
#             # neg_indicators = (neg_R_sorted * neg_sorted_indicators) ** alpha
#             neg_indicators = neg_sorted_indicators
#             sum_neg = torch.sum(neg_indicators, dim=2, keepdim=True)
#             cumsum_neg = torch.cumsum(neg_indicators, dim=2)
#             # step_cdf_neg = torch.zeros_like(cumsum_neg)
#             valid_neg = sum_neg > 1e-8
#             # step_cdf_neg[valid_neg] = cumsum_neg[valid_neg] / sum_neg[valid_neg]
#             step_cdf_neg = torch.where(
#                 valid_neg,  # 条件掩码（自动广播）
#                 cumsum_neg / sum_neg,  # 真值计算
#                 torch.zeros_like(cumsum_neg)  # 假值填充
#             )
#
#             # 计算富集分数
#             diff = (step_cdf_pos - step_cdf_neg) / num_genes
#             # 找到最大偏移值（绝对值最大的差值）
#             abs_diff = torch.abs(diff)
#             max_offset, max_offset_index = torch.max(abs_diff, dim=2)  # 沿基因维度找最大值
#             sign_offset = torch.gather(torch.sign(diff), dim=2, index=max_offset_index.unsqueeze(2)).squeeze(2)
#
#             # # 计算带符号的最大偏移值
#             es_scores = max_offset * sign_offset  # (batch_size, num_sets)
#             # es_scores = torch.sum(diff, dim=2)
#
#         else:
#
#
#             if self.thersold is not None:
#                 pos_indicators = indicators > self.thersold
#                 sorted_indicators = torch.gather(
#                     pos_indicators.unsqueeze(0).expand(batch_size, -1, -1),
#                     dim=2,
#                     index=S.unsqueeze(1).expand(-1, indicators.size(0), -1)
#                 )  # (batch_size, num_sets, num_genes)
#
#
#             else:
#                 sorted_indicators = torch.gather(
#                     indicators.unsqueeze(0).expand(batch_size, -1, -1),
#                     dim=2,
#                     index=S.unsqueeze(1).expand(-1, indicators.size(0), -1)
#                 )  # (batch_size, num_sets, num_genes)
#
#             R_sorted = torch.gather(R.unsqueeze(1).expand(-1, indicators.size(0), -1),
#                                     dim=2,
#                                     index=S.unsqueeze(1).expand(-1, indicators.size(0), -1))
#
#             # 计算正集加权 (alpha=0.25)
#             alpha = 0.25
#             weighted_pos = (R_sorted * sorted_indicators) ** alpha
#             sum_weighted_pos = torch.sum(weighted_pos, dim=2, keepdim=True)  # (batch_size, num_sets, 1)
#
#             # 计算正集累积分布 (避免除零)
#             cumsum_pos = torch.cumsum(weighted_pos, dim=2)
#             # step_cdf_pos = torch.zeros_like(cumsum_pos)
#             valid_pos = sum_weighted_pos > 1e-8
#             # step_cdf_pos[valid_pos] = cumsum_pos[valid_pos] / sum_weighted_pos[valid_pos]
#             step_cdf_pos = torch.where(
#                 valid_pos,  # 条件掩码（自动广播）
#                 cumsum_pos / sum_weighted_pos,  # 真值计算
#                 torch.zeros_like(cumsum_pos)  # 假值填充
#             )
#             # 计算负集累积分布
#             # neg_indicators = 1.0 - sorted_indicators
#             if self.thersold is not None:
#                 neg_indicators = sorted_indicators < self.thersold
#             else:
#                 # neg_indicators = 1.0 - sorted_indicators
#                 neg_indicators = sorted_indicators<0.1
#             sum_neg = torch.sum(neg_indicators, dim=2, keepdim=True)
#             cumsum_neg = torch.cumsum(neg_indicators, dim=2)
#             # step_cdf_neg = torch.zeros_like(cumsum_neg)
#             valid_neg = sum_neg > 1e-8
#             # step_cdf_neg[valid_neg] = cumsum_neg[valid_neg] / sum_neg[valid_neg]
#             step_cdf_neg = torch.where(
#                 valid_neg,  # 条件掩码（自动广播）
#                 cumsum_neg / sum_neg,  # 真值计算
#                 torch.zeros_like(cumsum_neg)  # 假值填充
#             )
#
#             # 计算富集分数
#             diff = (step_cdf_pos - step_cdf_neg) / num_genes
#             # 找到最大偏移值（绝对值最大的差值）
#             # abs_diff = torch.abs(diff)
#             # max_offset, max_offset_index = torch.max(abs_diff, dim=2)  # 沿基因维度找最大值
#             # sign_offset = torch.gather(torch.sign(diff), dim=2, index=max_offset_index.unsqueeze(2)).squeeze(2)
#             #
#             # # 计算带符号的最大偏移值
#             # es_scores = max_offset * sign_offset  # (batch_size, num_sets)
#             es_scores = torch.sum(diff, dim=2)
#
#         return es_scores


class GeneAttention(nn.Module):
    def __init__(self, num_genes, num_sets, key_dim=16, num_heads=4):
        super().__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads

        # 基因特征编码器（替代原MLP）
        self.gene_encoder = nn.Sequential(
            nn.Linear(1, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim)
        )

        # 多头注意力机制
        self.query = nn.Parameter(torch.randn(num_sets, key_dim))
        self.head_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(key_dim, key_dim),
                nn.ReLU()
            ) for _ in range(num_heads)
        ])

        # 残差连接
        self.residual = nn.Linear(1, num_sets)

    def forward(self, x):
        """x形状: (batch_size, num_genes, 1)"""
        # 基因特征编码
        gene_features = self.gene_encoder(x)  # (batch, genes, key_dim)

        # 多头注意力计算
        all_heads = []
        for head in self.head_proj:
            projected = head(gene_features)  # 各头独立变换
            # 计算基因集查询向量与基因键向量的相似度
            weights = torch.einsum('bgk,sk->bsg', projected, self.query)
            all_heads.append(weights)

        # 多头融合
        multi_head = torch.stack(all_heads, dim=1)  # (batch, heads, sets, genes)
        attention_weights = multi_head.mean(dim=1)  # 平均多头结果

        # 残差连接保留原始表达信息
        residual = self.residual(x).permute(0, 2, 1)  # (batch, sets, genes)
        return torch.sigmoid(attention_weights + residual)  # (batch, sets, genes)

class FeatureAttention(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(FeatureAttention, self).__init__()
        reduced_dim = max(input_dim // reduction, 4)  # 确保维度不小于4
        self.attention = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, input_dim),
            nn.Sigmoid()  # 生成0-1的注意力权重
        )

    def forward(self, x):
        att_weights = self.attention(x)  # 计算特征重要性权重
        return x * att_weights  # 特征重加权

