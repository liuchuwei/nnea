import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.train_utils import BuildGenesetLayer, BuildDeepGenesetLayer, BuildClassifier


class nnea(nn.Module):

    def __init__(self, config, loader):
        super().__init__()

        self.config = config
        self.gene = loader.gene
        self.sample_ids = loader.sample_ids
        self.config['num_genes'] = len(self.gene)

        # 先验知识处理
        self.prior_mask = self.load_gmt(
                    config['piror_knowledge'],
                    config['gene_names']) if config.get('use_piror_knowldege') else None


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

        if self.config['geneset_layer_mode'] == 'one_mode':
            self.gene_set_layer = BuildGenesetLayer(config=self.config, prior_tensor=self.prior_mask)
        elif self.config['geneset_layer_mode'] == 'deep_mode':
            self.gene_set_layers = BuildDeepGenesetLayer(config=self.config, prior_tensor=self.prior_mask)
    def build_classifier_layer(self):

        if self.config['geneset_layer_mode'] == 'deep_mode':
            # 在深度模式下使用所有层的特征拼接
            input_dim = self.config['geneset_layers'] * self.config['num_sets']
        elif self.config['geneset_layer_mode'] == 'one_mode':
            input_dim = self.config['num_sets']

        self.classifier = BuildClassifier(self.config, input_dim=input_dim)

    def build_decoder(self):
        """构建自编码器的解码器层"""
        # 使用基因集层相同的输入维度
        if self.config['geneset_layer_mode'] == 'deep_mode':
            decoder_input_dim = self.config['geneset_layers'] * self.config['num_sets']
        else:
            decoder_input_dim = self.config['num_sets']

        # 解码器：从基因集分数重构基因表达
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, self.config['num_genes'])
        )

    def build_model(self):

        self.build_geneset_layer()

        if self.config['task'] == "autoencoder":
            self.build_decoder()
        else:
            self.build_classifier_layer()

    def regularization_loss(self):

        """返回基因集层的正则化损失"""
        if self.config['geneset_layer_mode'] == 'deep_mode':
            total_loss = 0
            for layer in self.gene_set_layers:
                total_loss += layer.regularization_loss()
            return total_loss
        else:
            return self.gene_set_layer.regularization_loss()

    def forward(self, R, S):

        if self.config['geneset_layer_mode'] == 'deep_mode':
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
                # 渐进式衰减策略（核心修改）
                decay_factor = torch.sigmoid(-self.config.get('decay_factor', 5.0) *
                                             (max_indicators - self.config.get('decay_threshold', 0.1)))
                # 更新全局掩码（新激活的基因会获得更小的衰减系数）
                gene_mask = gene_mask * decay_factor
                # 保护机制：防止所有基因被过度抑制
                if torch.sum(gene_mask) < 1e-3:  # 总和阈值
                    gene_mask = torch.ones_like(gene_mask) * 0.5  # 部分重置

                # new_mask = (max_indicators < 0.1).float()  # 指示值>0.1的基因设为0
                # if torch.sum(new_mask) == 0:  # 避免全零掩码
                #     new_mask = torch.ones_like(new_mask)
                # gene_mask = gene_mask * new_mask  # 更新全局mask

            # 拼接所有层的富集分数
            es_scores = torch.cat(all_es_scores, dim=1)
        else:
            # 单层模式
            es_scores = self.gene_set_layer(R, S)

        # 预测类别
        if self.config['task'] in ["regression", "classification", "cox", "umap"]:
            logits = self.classifier(es_scores)

            if self.config['task'] in ['classification']:
                res = F.log_softmax(logits, dim=1)
            elif self.config['task'] in ['regression', "cox", "umap"]:
                res = logits

            return res

        elif self.config['task'] == "autoencoder":
            return self.decoder(es_scores)
