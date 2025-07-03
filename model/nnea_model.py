import torch
import torch.nn as nn
from model.nnea_layers import TrainableGeneSetLayer, FeatureAttention, AttentionClassifier
import torch.nn.functional as F


class nnea(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.mode = config.get('mode', 'deep_mode')
        self.num_layers = config.get('num_layers', 3) if self.mode == 'deep_mode' else 1
        # 先验知识处理
        self.prior_mask = self.load_gmt(
                    config['piror_knowledge'],
                    config['gene_names']) if config.get('piror_knowledge') else None

    def load_gmt(self, gmt_path, gene_names):
        """ 解析.gmt文件为基因集指示矩阵 """
        pathways = []
        with open(gmt_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) < 3: continue
                pathway_genes = items[2:]
                pathways.append(pathway_genes)

        # 构建 (num_pathways, num_genes) 二值矩阵
        indicator = torch.zeros(len(pathways), len(gene_names))
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

        for p_idx, genes in enumerate(pathways):
            for gene in genes:
                if gene in gene_to_idx:
                    indicator[p_idx, gene_to_idx[gene]] = 1
        return indicator

    def build_geneset_layer(self):


        if self.mode == 'deep_mode':

            self.gene_set_layers = nn.ModuleList()
            for i in range(self.num_layers):
                if i == 0 and self.prior_knowledge is not None:
                    prior_tensor = self.prior_knowledge.to(self.config.get('device', 'cpu'))
                    layer = TrainableGeneSetLayer(
                        num_genes=self.config['num_genes'],
                        num_sets=prior_tensor.shape[0],
                        min_set_size=self.config['set_min_size'],
                        max_set_size=self.config['set_max_size'],
                        alpha=self.config['alpha'],
                        is_deep_layer=True,
                        layer_index=i,
                        prior_knowledge=prior_tensor,
                        freeze_prior=self.config['freeze_prior']
                    )
                    layer.set_membership.requires_grad = False  # 冻结参数
                else:
                    layer = TrainableGeneSetLayer(
                        num_genes=self.config['num_genes'],
                        num_sets=self.config['num_sub_sets'] if self.mode == 'deep_mode' else self.config['num_sets'],
                        min_set_size=self.config['set_min_size'],
                        max_set_size=self.config['set_max_size'],
                        alpha=self.config['alpha'],
                        is_deep_layer=(self.mode == 'deep_mode'),
                        layer_index=i
                    )
                self.gene_set_layers.append(layer)

        else:  # one_mode
            if self.prior_knowledge is not None:
                prior_tensor = self.prior_knowledge.to(self.config.get('device', 'cpu'))
                self.gene_set_layer = TrainableGeneSetLayer(
                    num_genes=self.config['num_genes'],
                    num_sets=prior_tensor.shape[0],
                    min_set_size=self.config['set_min_size'],
                    max_set_size=self.config['set_max_size'],
                    alpha=self.config['alpha'],
                    is_deep_layer=False,
                    prior_knowledge=prior_tensor,
                    freeze_prior=self.config['freeze_prior']
                )
            else:
                self.gene_set_layer = TrainableGeneSetLayer(
                    num_genes=self.config['num_genes'],
                    num_sets=self.config['num_sets'],
                    min_set_size=self.config['set_min_size'],
                    max_set_size=self.config['set_max_size'],
                    alpha=self.config['alpha'],
                    is_deep_layer=False
                )

    def build_classifier_layer(self):

        if self.mode == 'deep_mode':
            # 在深度模式下使用所有层的特征拼接
            total_sets = self.num_layers * self.config['num_sub_sets']
        else:
            total_sets = self.config['num_sets']

        if self.config['classifier'] == "linear":
            self.classifier = nn.Sequential(
                nn.Linear(total_sets, self.config['hidden_dim']),
                nn.ReLU(),
                nn.BatchNorm1d(self.config['hidden_dim']),
                nn.Dropout(0.4),
                nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'] // 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.config['hidden_dim'] // 2),
                nn.Dropout(0.3),
                nn.Linear(self.config['hidden_dim'] // 2, self.config['num_classes'])
            )

        elif self.config['classifier'] == "attention":

            self.classifier =AttentionClassifier(
            num_sets=self.config['num_sets'],
            num_classes=self.config['num_classes'],
            hidden_dim=self.config.get('attn_hidden_dim', 64)  # 可配置的隐藏层维度
        )

    def build_model(self):

        self.build_geneset_layer()
        self.build_classifier_layer()

    def regularization_loss(self):

        """返回基因集层的正则化损失"""
        if self.mode == 'deep_mode':
            total_loss = 0
            for layer in self.gene_set_layers:
                total_loss += layer.regularization_loss()
            return total_loss
        else:
            return self.gene_set_layer.regularization_loss()

    def forward(self, R, S):

        if self.mode == 'deep_mode':
            # 深度模式：分层计算
            all_es_scores = []
            gene_mask = torch.ones(R.shape[1], device=R.device)

            for layer in self.gene_set_layers:
                # 应用当前层的mask
                R_masked = R * gene_mask
                es_scores = layer(R_masked, S, gene_mask)
                all_es_scores.append(es_scores)

                # 更新基因mask - 将当前层激活的基因设置为0
                indicators = layer.get_set_indicators(R_masked).detach()
                max_indicators, _ = torch.max(indicators, dim=0)  # 获取每个基因的最大指示值
                new_mask = (max_indicators < 0.1).float()  # 指示值>0.1的基因设为0
                if torch.sum(new_mask) == 0:  # 避免全零掩码
                    new_mask = torch.ones_like(new_mask)
                gene_mask = gene_mask * new_mask  # 更新全局mask

            # 拼接所有层的富集分数
            es_scores = torch.cat(all_es_scores, dim=1)
        else:
            # 单层模式
            es_scores = self.gene_set_layer(R, S)
        # 预测类别
        if self.config['classifier'] == "linear":
            logits = self.classifier(es_scores)
        elif self.config['classifier'] == "attention":
            logits, attention_weights = self.classifier(es_scores)


        return F.log_softmax(logits, dim=1)
