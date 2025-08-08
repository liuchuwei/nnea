import os
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from nnea.model.base import BaseModel
import warnings
warnings.filterwarnings('ignore')
import random
import logging

class NeuralUMAP(nn.Module):
    """
    基于神经网络的UMAP实现
    
    这个实现使用编码器-解码器架构来学习高维数据到低维空间的映射，
    同时保持数据的局部和全局结构。
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 2, 
                 hidden_dims: List[int] = [128, 64, 32], 
                 dropout: float = 0.1):
        """
        初始化神经网络UMAP模型
        
        Args:
            input_dim: 输入数据维度
            embedding_dim: 嵌入空间维度（通常为2用于可视化）
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比率
        """
        super(NeuralUMAP, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        
        # 编码器：高维 -> 低维
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器：低维 -> 高维（可选，用于重构）
        decoder_layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 重构输出层
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """前向传播"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        """仅编码"""
        return self.encoder(x)
    
    def decode(self, z):
        """仅解码"""
        return self.decoder(z)


class UMAPLoss(nn.Module):
    """
    UMAP损失函数实现，使用PCA寻找最近邻对（优化版本）
    
    参考nn_umap.py的实现，添加缓存机制提升训练效率
    """
    
    def __init__(self, min_dist: float = 0.1, a: float = 1.0, b: float = 1.0, 
                 n_neighbors: int = 15, pca_components: int = 50, use_vectorized: bool = True, debug: bool = False):
        """
        初始化UMAP损失函数
        
        Args:
            min_dist: 最小距离参数
            a, b: UMAP的a和b参数
            n_neighbors: 邻居数量
            pca_components: PCA组件数量
            use_vectorized: 是否使用向量化实现（更高效）
            debug: 是否启用调试模式
        """
        super(UMAPLoss, self).__init__()
        self.min_dist = min_dist
        self.a = a
        self.b = b
        self.n_neighbors = n_neighbors
        self.pca_components = pca_components
        self.use_vectorized = use_vectorized
        self.debug = debug
        self.pca = None
        self.original_data = None
        # 添加缓存变量
        self.positive_pairs = None
        self.negative_pairs = None
        self.is_fitted = False
        # 添加全局距离矩阵缓存
        self.global_distances = None
        # 添加损失统计信息
        self.loss_stats = {}
        # 添加logger
        import logging
        self.logger = logging.getLogger(__name__)
        
    def set_umap_params(self, a: float = None, b: float = None, min_dist: float = None):
        """
        设置UMAP参数
        
        Args:
            a: UMAP的a参数
            b: UMAP的b参数
            min_dist: 最小距离参数
        """
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if min_dist is not None:
            self.min_dist = min_dist
        
        self.logger.info(f"UMAP参数已更新: a={self.a}, b={self.b}, min_dist={self.min_dist}")
        
    def get_loss_stats(self):
        """
        获取损失函数统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'a': self.a,
            'b': self.b,
            'min_dist': self.min_dist,
            'n_neighbors': self.n_neighbors,
            'pca_components': self.pca_components,
            'is_fitted': self.is_fitted,
            'has_global_distances': hasattr(self, 'global_distances') and self.global_distances is not None
        }
        
        if self.is_fitted:
            stats.update({
                'nbr_indices_shape': self.nbr_indices.shape if self.nbr_indices is not None else None,
                'original_data_shape': self.original_data.shape if self.original_data is not None else None,
                'global_distances_shape': self.global_distances.shape if self.global_distances is not None else None
            })
        
        return stats
        
    def fit_pca(self, X: np.ndarray, nadata=None):
        """
        使用PCA拟合数据并返回正样本和负样本索引（只计算一次）
        
        Args:
            X: 原始高维数据
            nadata: nadata对象，如果提供且包含预计算的PCA数据，则直接使用
            
        Returns:
            tuple: (pos_indices, neg_indices) 正样本和负样本索引数组
        """
        if self.is_fitted:
            return self.pos_indices, self.neg_indices
            
        # 检查是否可以从nadata.uns中读取预计算的PCA数据
        X_pca = None
        
        if nadata is not None and hasattr(nadata, 'uns'):
            # 检查是否有预计算的PCA数据
            if 'pca' in nadata.uns:
                X_pca = nadata.uns['pca']
                self.logger.info("从nadata.uns中读取预计算的PCA数据")
                
                # 确保PCA数据的形状正确
                if X_pca.shape[0] != X.shape[0]:
                    self.logger.warning(f"PCA数据样本数({X_pca.shape[0]})与输入数据样本数({X.shape[0]})不匹配，重新计算")
                    X_pca = None
            else:
                self.logger.info("nadata.uns中未找到预计算的PCA数据")
        else:
            self.logger.info("nadata对象为空或没有uns属性")
            
        # 如果没有预计算的PCA数据，则重新计算
        if X_pca is None:
            # 限制PCA组件数量不超过特征数量
            n_components = min(self.pca_components, X.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X)
            
            # 使用PCA降维
            X_pca = self.pca.transform(X)
            self.logger.info(f"重新计算PCA，组件数: {n_components}")
        else:
            self.logger.info(f"使用预计算的PCA数据，形状: {X_pca.shape}")
            # 创建一个虚拟的PCA对象以保持兼容性
            self.pca = None
        
        # 使用sklearn的NearestNeighbors寻找最近邻
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='auto').fit(X_pca)
        distances, nbr_indices = nbrs.kneighbors(X_pca)
        
        # 计算并缓存全局距离矩阵（用于负样本选择）
        self.global_distances = distances
        
        # 生成正样本和负样本索引
        pos_indices, neg_indices = self._generate_pos_neg_pairs(nbr_indices, X.shape[0])
        
        # 将结果存储到nadata.uns中
        if nadata is not None:
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            nadata.uns['pca'] = X_pca
            nadata.uns['nbr_indices'] = nbr_indices
            nadata.uns['pos_indices'] = pos_indices
            nadata.uns['neg_indices'] = neg_indices
            nadata.uns['global_distances'] = self.global_distances
            self.logger.info("已将PCA、pos_indices、neg_indices和global_distances数据存储到nadata.uns中")
        
        # 缓存结果
        self.nbr_indices = nbr_indices
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.original_data = X
        self.is_fitted = True
        
        return pos_indices, neg_indices
    
    def find_neighbors_pca(self, X: np.ndarray, nadata=None) -> np.ndarray:
        """
        获取已缓存的近邻索引（如果未缓存则先计算）
        
        Args:
            X: 原始高维数据
            nadata: nadata对象，可选
            
        Returns:
            nbr_indices: 近邻索引数组
        """
        if not self.is_fitted:
            self.fit_pca(X, nadata)
        
        return self.nbr_indices
    
    def _generate_pos_neg_pairs(self, nbr_indices: np.ndarray, n_samples: int):
        """
        生成正样本和负样本对
        
        Args:
            nbr_indices: 近邻索引数组，形状为(n_samples, n_neighbors+1)
            n_samples: 样本数量
            
        Returns:
            tuple: (pos_indices, neg_indices) 正样本和负样本索引
        """
        # 正样本：每个样本与其近邻（排除自身）
        pos_pairs = []
        for i in range(n_samples):
            # 获取当前样本的近邻（排除自身）
            neighbors = nbr_indices[i][1:]  # 排除第一个（自身）
            for neighbor in neighbors:
                pos_pairs.append([i, neighbor])
        
        pos_indices = np.array(pos_pairs)
        
        # 负样本：随机选择非近邻的样本对
        neg_pairs = []
        n_neg_per_sample = min(self.n_neighbors, n_samples - self.n_neighbors - 1)  # 确保不超过可用样本数
        
        for i in range(n_samples):
            # 获取当前样本的近邻
            neighbors = set(nbr_indices[i])
            
            # 随机选择非近邻的样本作为负样本
            non_neighbors = [j for j in range(n_samples) if j not in neighbors and j != i]
            
            if len(non_neighbors) > 0:
                # 随机选择负样本
                n_neg = min(n_neg_per_sample, len(non_neighbors))
                selected_neg = np.random.choice(non_neighbors, size=n_neg, replace=False)
                
                for neg_idx in selected_neg:
                    neg_pairs.append([i, neg_idx])
        
        neg_indices = np.array(neg_pairs) if neg_pairs else np.empty((0, 2), dtype=int)
        
        self.logger.info(f"生成样本对: 正样本 {len(pos_indices)} 对, 负样本 {len(neg_indices)} 对")
        
        return pos_indices, neg_indices

    def forward(self, embeddings, original_data=None, nadata=None, batch_pos_indices=None, batch_neg_indices=None):
        """
        计算UMAP损失
        
        Args:
            embeddings: 嵌入向量，形状为(batch_size, embedding_dim)
            original_data: 原始数据，用于计算重构损失
            nadata: nadata对象
            batch_pos_indices: 批次正样本索引，形状为(batch_size, n_pos_pairs, 2)
            batch_neg_indices: 批次负样本索引，形状为(batch_size, n_neg_pairs, 2)
            
        Returns:
            total_loss: 总损失
        """
        if batch_pos_indices is None or batch_neg_indices is None:
            self.logger.warning("未提供正样本或负样本索引，返回零损失")
            return torch.tensor(0.0, device=embeddings.device)
        
        # 验证输入形状
        batch_size = embeddings.shape[0]
        if batch_pos_indices.shape[0] != batch_size or batch_neg_indices.shape[0] != batch_size:
            self.logger.error(f"批次大小不匹配: embeddings={batch_size}, pos_indices={batch_pos_indices.shape[0]}, neg_indices={batch_neg_indices.shape[0]}")
            return torch.tensor(0.0, device=embeddings.device)
        
        # 确保索引是torch张量并移动到正确的设备
        if not torch.is_tensor(batch_pos_indices):
            batch_pos_indices = torch.tensor(batch_pos_indices, dtype=torch.long, device=embeddings.device)
        elif batch_pos_indices.device != embeddings.device:
            batch_pos_indices = batch_pos_indices.to(embeddings.device)
            
        if not torch.is_tensor(batch_neg_indices):
            batch_neg_indices = torch.tensor(batch_neg_indices, dtype=torch.long, device=embeddings.device)
        elif batch_neg_indices.device != embeddings.device:
            batch_neg_indices = batch_neg_indices.to(embeddings.device)
        
        # 选择使用向量化实现还是循环实现
        if self.use_vectorized:
            # 使用向量化实现（更高效）
            pos_loss, neg_loss = self._compute_loss_vectorized(embeddings, batch_pos_indices, batch_neg_indices)
        else:
            # 使用循环实现（更直观）
            pos_loss = self._compute_positive_loss(embeddings, batch_pos_indices)
            neg_loss = self._compute_negative_loss(embeddings, batch_neg_indices)
        
        # 总损失
        total_loss = pos_loss + neg_loss
        
        # 记录损失统计信息
        if hasattr(self, 'loss_stats'):
            self.loss_stats['pos_loss'] = pos_loss.item()
            self.loss_stats['neg_loss'] = neg_loss.item()
            self.loss_stats['total_loss'] = total_loss.item()
        
        # 调试信息（仅在需要时显示）
        if self.debug:
            self.logger.info(f"UMAP损失 - 正样本: {pos_loss.item():.6f}, 负样本: {neg_loss.item():.6f}, 总计: {total_loss.item():.6f}")
        
        return total_loss
    
    def _compute_positive_loss(self, embeddings, pos_indices):
        """
        计算正样本损失
        
        Args:
            embeddings: 嵌入向量，形状为(batch_size, embedding_dim)
            pos_indices: 正样本索引对，形状为(batch_size, n_pos_pairs, 2)
            
        Returns:
            pos_loss: 正样本损失
        """
        if pos_indices.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # 处理批次数据
        batch_size = embeddings.shape[0]
        pos_pairs = []
        
        # 遍历每个样本的正样本对
        for i in range(batch_size):
            sample_pos_pairs = pos_indices[i]  # 形状为(n_pos_pairs, 2)
            for pair in sample_pos_pairs:
                idx1, idx2 = pair[0].item(), pair[1].item()
                # 确保索引在批次范围内
                if 0 <= idx1 < batch_size and 0 <= idx2 < batch_size:
                    pos_pairs.append([embeddings[idx1], embeddings[idx2]])
        
        if not pos_pairs:
            return torch.tensor(0.0, device=embeddings.device)
        
        # 使用更高效的方式堆叠张量
        pos_pairs = torch.stack([torch.stack(pair) for pair in pos_pairs])
        
        # 计算正样本对之间的距离
        pos_distances = torch.norm(pos_pairs[:, 0] - pos_pairs[:, 1], dim=1)
        
        # UMAP正样本损失：使用交叉熵损失
        # 目标：正样本对应该接近
        pos_targets = torch.ones(len(pos_distances), device=embeddings.device)
        
        # 使用sigmoid将距离转换为概率
        pos_probs = torch.sigmoid(-pos_distances / self.min_dist)
        pos_loss = F.binary_cross_entropy(pos_probs, pos_targets)
        
        return pos_loss
    
    def _compute_negative_loss(self, embeddings, neg_indices):
        """
        计算负样本损失
        
        Args:
            embeddings: 嵌入向量，形状为(batch_size, embedding_dim)
            neg_indices: 负样本索引对，形状为(batch_size, n_neg_pairs, 2)
            
        Returns:
            neg_loss: 负样本损失
        """
        if neg_indices.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # 处理批次数据
        batch_size = embeddings.shape[0]
        neg_pairs = []
        
        # 遍历每个样本的负样本对
        for i in range(batch_size):
            sample_neg_pairs = neg_indices[i]  # 形状为(n_neg_pairs, 2)
            for pair in sample_neg_pairs:
                idx1, idx2 = pair[0].item(), pair[1].item()
                # 确保索引在批次范围内
                if 0 <= idx1 < batch_size and 0 <= idx2 < batch_size:
                    neg_pairs.append([embeddings[idx1], embeddings[idx2]])
        
        if not neg_pairs:
            return torch.tensor(0.0, device=embeddings.device)
        
        # 使用更高效的方式堆叠张量
        neg_pairs = torch.stack([torch.stack(pair) for pair in neg_pairs])
        
        # 计算负样本对之间的距离
        neg_distances = torch.norm(neg_pairs[:, 0] - neg_pairs[:, 1], dim=1)
        
        # UMAP负样本损失：使用交叉熵损失
        # 目标：负样本对应该远离
        neg_targets = torch.zeros(len(neg_distances), device=embeddings.device)
        
        # 使用sigmoid将距离转换为概率
        neg_probs = torch.sigmoid(-neg_distances / self.min_dist)
        neg_loss = F.binary_cross_entropy(neg_probs, neg_targets)
        
        return neg_loss
    
    def _compute_loss_vectorized(self, embeddings, pos_indices, neg_indices):
        """
        向量化计算UMAP损失（更高效的实现）
        
        Args:
            embeddings: 嵌入向量，形状为(batch_size, embedding_dim)
            pos_indices: 正样本索引对，形状为(batch_size, n_pos_pairs, 2)
            neg_indices: 负样本索引对，形状为(batch_size, n_neg_pairs, 2)
            
        Returns:
            pos_loss, neg_loss: 正样本损失和负样本损失
        """
        batch_size = embeddings.shape[0]
        
        # 向量化处理正样本对
        pos_loss = torch.tensor(0.0, device=embeddings.device)
        if pos_indices.numel() > 0:
            # 重塑索引以便向量化处理
            pos_indices_flat = pos_indices.view(-1, 2)
            
            # 过滤有效的索引对（在批次范围内）
            valid_mask = (pos_indices_flat[:, 0] >= 0) & (pos_indices_flat[:, 0] < batch_size) & \
                        (pos_indices_flat[:, 1] >= 0) & (pos_indices_flat[:, 1] < batch_size)
            
            if valid_mask.any():
                valid_pos_indices = pos_indices_flat[valid_mask]
                pos_embeddings1 = embeddings[valid_pos_indices[:, 0]]
                pos_embeddings2 = embeddings[valid_pos_indices[:, 1]]
                
                # 计算距离
                pos_distances = torch.norm(pos_embeddings1 - pos_embeddings2, dim=1)
                
                # 计算损失
                pos_targets = torch.ones(len(pos_distances), device=embeddings.device)
                pos_probs = torch.sigmoid(-pos_distances / self.min_dist)
                pos_loss = F.binary_cross_entropy(pos_probs, pos_targets)
        
        # 向量化处理负样本对
        neg_loss = torch.tensor(0.0, device=embeddings.device)
        if neg_indices.numel() > 0:
            # 重塑索引以便向量化处理
            neg_indices_flat = neg_indices.view(-1, 2)
            
            # 过滤有效的索引对（在批次范围内）
            valid_mask = (neg_indices_flat[:, 0] >= 0) & (neg_indices_flat[:, 0] < batch_size) & \
                        (neg_indices_flat[:, 1] >= 0) & (neg_indices_flat[:, 1] < batch_size)
            
            if valid_mask.any():
                valid_neg_indices = neg_indices_flat[valid_mask]
                neg_embeddings1 = embeddings[valid_neg_indices[:, 0]]
                neg_embeddings2 = embeddings[valid_neg_indices[:, 1]]
                
                # 计算距离
                neg_distances = torch.norm(neg_embeddings1 - neg_embeddings2, dim=1)
                
                # 计算损失
                neg_targets = torch.zeros(len(neg_distances), device=embeddings.device)
                neg_probs = torch.sigmoid(-neg_distances / self.min_dist)
                neg_loss = F.binary_cross_entropy(neg_probs, neg_targets)
        
        return pos_loss, neg_loss


class NNEAUMAP(BaseModel):
    """
    NNEA UMAP降维器
    实现基于神经网络的UMAP降维，提供可解释的降维结果
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化NNEA UMAP降维器
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        self.task = 'umap'

    def build(self, nadata) -> None:
        """
        构建模型
        
        Args:
            nadata: nadata对象
        """
        if nadata is None:
            raise ValueError("nadata对象不能为空")
        
        # 获取输入维度
        if hasattr(nadata, 'X') and nadata.X is not None:
            input_dim = nadata.X.shape[1]  # 基因数量
        else:
            raise ValueError("表达矩阵未加载")
        
        # 获取UMAP配置
        umap_config = self.config.get('umap', {})
        embedding_dim = umap_config.get('embedding_dim', 2)
        hidden_dims = umap_config.get('hidden_dims', [128, 64, 32])
        dropout = umap_config.get('dropout', 0.1)
        
        # 更新配置
        self.config['input_dim'] = input_dim
        self.config['embedding_dim'] = embedding_dim
        self.config['device'] = str(self.device)
        
        # 创建模型
        self.model = NeuralUMAP(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        self.model.to(self.device)
        
        # 创建UMAP损失函数（使用优化的PCA版本）
        n_neighbors = umap_config.get('n_neighbors', 15)
        min_dist = umap_config.get('min_dist', 0.1)
        pca_components = umap_config.get('pca_components', 50)
        
        self.umap_loss = UMAPLoss(
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            pca_components=pca_components
        ).to(self.device)
        
        self.logger.info(f"NNEA UMAP降维器已构建: 输入维度={input_dim}, 嵌入维度={embedding_dim}")
        self.logger.info(f"隐藏层维度: {hidden_dims}")
        self.logger.info(f"UMAP参数: n_neighbors={n_neighbors}, min_dist={min_dist}, pca_components={pca_components}")
    
    def train(self, nadata, verbose: int = 1, max_epochs: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            nadata: nadata对象
            verbose: 详细程度
                0=只显示进度条
                1=显示训练损失
                2=显示训练损失和重构损失
            max_epochs: 最大训练轮数，如果为None则使用配置中的epochs
            **kwargs: 额外参数
            
        Returns:
            训练结果字典
        """
        if self.model is None:
            raise ValueError("模型未构建")
        
        # 准备数据
        X = nadata.X

        # 在训练开始前完成PCA和近邻计算（只计算一次）
        self.logger.info("正在计算PCA和近邻关系...")
        pos_indices, neg_indices = self.umap_loss.fit_pca(X, nadata)
        self.logger.info("PCA和近邻计算完成！")
        
        # 训练参数
        training_config = self.config.get('training', {})
        if max_epochs is None:
            epochs = training_config.get('epochs', 100)
        else:
            epochs = max_epochs
        learning_rate = training_config.get('learning_rate', 0.001)
        batch_size = training_config.get('batch_size', 32)
        test_size = training_config.get('test_size', 0.2)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X)
        
        # 自定义数据集类
        class UMAPDataset(torch.utils.data.Dataset):
            def __init__(self, X, pos_indices, neg_indices):
                self.X = X
                self.pos_indices = pos_indices
                self.neg_indices = neg_indices
                
                # 为每个样本创建索引映射
                self.sample_to_pos = {}
                self.sample_to_neg = {}
                
                # 构建样本到正样本对的映射
                for i, (idx1, idx2) in enumerate(pos_indices):
                    if idx1 not in self.sample_to_pos:
                        self.sample_to_pos[idx1] = []
                    self.sample_to_pos[idx1].append(idx2)
                    
                    if idx2 not in self.sample_to_pos:
                        self.sample_to_pos[idx2] = []
                    self.sample_to_pos[idx2].append(idx1)
                
                # 构建样本到负样本对的映射
                for i, (idx1, idx2) in enumerate(neg_indices):
                    if idx1 not in self.sample_to_neg:
                        self.sample_to_neg[idx1] = []
                    self.sample_to_neg[idx1].append(idx2)
                    
                    if idx2 not in self.sample_to_neg:
                        self.sample_to_neg[idx2] = []
                    self.sample_to_neg[idx2].append(idx1)
                
                # 计算最大正样本和负样本数量，用于填充
                self.max_pos_pairs = 0
                self.max_neg_pairs = 0
                for i in range(len(X)):
                    pos_count = len(self.sample_to_pos.get(i, []))
                    neg_count = len(self.sample_to_neg.get(i, []))
                    self.max_pos_pairs = max(self.max_pos_pairs, pos_count)
                    self.max_neg_pairs = max(self.max_neg_pairs, neg_count)
                
                # 确保至少有一个样本对
                self.max_pos_pairs = max(self.max_pos_pairs, 1)
                self.max_neg_pairs = max(self.max_neg_pairs, 1)
                
                # 记录最大样本对数量
                print(f"数据集统计: 最大正样本对数量={self.max_pos_pairs}, 最大负样本对数量={self.max_neg_pairs}")
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                # 获取当前样本的正样本索引
                pos_neighbors = self.sample_to_pos.get(idx, [])
                if len(pos_neighbors) == 0:
                    pos_neighbors = [idx]  # 如果没有正样本，使用自身
                
                # 获取当前样本的负样本索引
                neg_neighbors = self.sample_to_neg.get(idx, [])
                if len(neg_neighbors) == 0:
                    neg_neighbors = [idx]  # 如果没有负样本，使用自身
                
                # 创建正样本对索引并填充到固定大小
                pos_pairs = [[idx, neighbor] for neighbor in pos_neighbors]
                while len(pos_pairs) < self.max_pos_pairs:
                    pos_pairs.append([idx, idx])  # 用自身填充
                
                # 创建负样本对索引并填充到固定大小
                neg_pairs = [[idx, neighbor] for neighbor in neg_neighbors]
                while len(neg_pairs) < self.max_neg_pairs:
                    neg_pairs.append([idx, idx])  # 用自身填充
                
                # 将原始索引转换为批次内索引（相对位置）
                # 这里我们返回原始索引，在DataLoader的collate_fn中进行转换
                return (self.X[idx], 
                       torch.tensor(pos_pairs, dtype=torch.long),
                       torch.tensor(neg_pairs, dtype=torch.long))
        
        # 构建完整数据集
        full_dataset = UMAPDataset(X_tensor, pos_indices, neg_indices)
        
        # 使用random_split分割数据集
        n_samples = X.shape[0]
        train_size = int(n_samples * (1 - test_size))
        test_size_split = n_samples - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size_split]
        )
        
        self.logger.info(f"数据划分: 训练集 {len(train_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
        
        # 存储训练集索引供后续使用（从random_split获取）
        self.train_indices = train_dataset.indices
        
        # 定义collate函数，将原始索引转换为批次内索引
        def umap_collate_fn(batch):
            """
            将批次数据中的原始索引转换为批次内索引
            """
            batch_X = []
            batch_pos_indices = []
            batch_neg_indices = []
            
            # 获取当前批次中所有样本在原始数据集中的索引
            # 由于random_split会重新索引，我们需要从train_dataset.indices获取原始索引
            batch_original_indices = [train_dataset.indices[i] for i in range(len(batch))]
            
            # 创建原始索引到批次内索引的映射
            original_to_batch = {orig_idx: batch_idx for batch_idx, orig_idx in enumerate(batch_original_indices)}
            
            for i, (X_item, pos_pairs, neg_pairs) in enumerate(batch):
                batch_X.append(X_item)
                
                # 将原始索引转换为批次内索引
                pos_pairs_batch = pos_pairs.clone()
                neg_pairs_batch = neg_pairs.clone()
                
                # 转换正样本对索引
                for j in range(pos_pairs_batch.shape[0]):
                    orig_idx1, orig_idx2 = pos_pairs_batch[j]
                    if orig_idx1 in original_to_batch and orig_idx2 in original_to_batch:
                        pos_pairs_batch[j, 0] = original_to_batch[orig_idx1]
                        pos_pairs_batch[j, 1] = original_to_batch[orig_idx2]
                    else:
                        # 如果索引不在当前批次中，使用自身索引
                        pos_pairs_batch[j, 0] = i
                        pos_pairs_batch[j, 1] = i
                
                # 转换负样本对索引
                for j in range(neg_pairs_batch.shape[0]):
                    orig_idx1, orig_idx2 = neg_pairs_batch[j]
                    if orig_idx1 in original_to_batch and orig_idx2 in original_to_batch:
                        neg_pairs_batch[j, 0] = original_to_batch[orig_idx1]
                        neg_pairs_batch[j, 1] = original_to_batch[orig_idx2]
                    else:
                        # 如果索引不在当前批次中，使用自身索引
                        neg_pairs_batch[j, 0] = i
                        neg_pairs_batch[j, 1] = i
                
                batch_pos_indices.append(pos_pairs_batch)
                batch_neg_indices.append(neg_pairs_batch)
            
            # 堆叠批次数据
            batch_X = torch.stack(batch_X)
            batch_pos_indices = torch.stack(batch_pos_indices)
            batch_neg_indices = torch.stack(batch_neg_indices)
            
            return batch_X, batch_pos_indices, batch_neg_indices
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=umap_collate_fn
        )
        
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 早停机制参数
        patience = training_config.get('patience', 10)
        min_delta = 1e-6
        
        # 早停变量初始化
        best_loss = float('inf')
        patience_counter = 0
        early_stopped = False
        
        # 训练循环
        train_losses = []
        
        if verbose >= 1:
            self.logger.info("开始训练NNEA UMAP模型...")
            self.logger.info(f"早停配置: patience={patience}, min_delta={min_delta}")
        
        # 导入tqdm用于进度条
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        # 创建进度条（只有verbose=0时显示）
        if verbose == 0 and use_tqdm:
            pbar = tqdm(range(epochs), desc="训练进度")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # 训练模式
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # 使用数据加载器进行批处理训练
            for batch_idx, (batch_X, batch_pos_indices, batch_neg_indices) in enumerate(train_loader):

                # 将数据移动到设备
                batch_X = batch_X.to(self.device)
                
                optimizer.zero_grad()
                
                try:
                    # 前向传播
                    encoded, decoded = self.model(batch_X)

                    # 调试信息：显示indices数量
                    if verbose >= 2 and batch_idx == 0:
                        self.logger.info(f"Epoch {epoch}, Batch {batch_idx}: batch_pos_indices={batch_pos_indices.shape}, batch_neg_indices={batch_neg_indices.shape}")


                    umap_loss = self.umap_loss(encoded, X, nadata, batch_pos_indices, batch_neg_indices)
                    
                    # 计算重构损失（可选）
                    recon_loss = F.mse_loss(decoded, batch_X)
                    
                    # 总损失
                    total_loss = umap_loss + 0.1 * recon_loss
                    
                    # 反向传播
                    total_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: 训练过程中出现错误: {e}")
                    continue
            
            # 计算平均损失
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                train_losses.append(avg_loss)
                
                # verbose=1时显示训练损失
                if verbose >= 1:
                    self.logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.4f}")
                
                # 早停检查
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 检查是否触发早停
                if patience_counter >= patience:
                    early_stopped = True
                    self.logger.info(f"🛑 Epoch {epoch}: 触发早停！损失在{patience}个epoch内未改善")
                    break
        
        # 训练完成
        self.is_trained = True
        
        # 记录早停信息
        if early_stopped:
            self.logger.info(f"📊 训练因早停而结束，实际训练了{epoch+1}个epoch")
        else:
            self.logger.info(f"📊 训练完成，共训练了{epochs}个epoch")
        
        # 显示缓存信息
        cache_info = self.get_cache_info()
        self.logger.info("缓存信息：")
        self.logger.info(f"- PCA已拟合: {cache_info['pca_fitted']}")
        self.logger.info(f"- 近邻索引形状: {cache_info['nbr_indices_shape']}")
        self.logger.info(f"- PCA组件数: {cache_info['pca_components']}")
        self.logger.info(f"- 原始数据形状: {cache_info['original_data_shape']}")
        self.logger.info(f"- 全局距离矩阵形状: {cache_info['global_distances_shape']}")
        self.logger.info(f"- 智能负样本采样: {cache_info['smart_negative_sampling']}")
        self.logger.info("✅ 改进：使用全局距离信息进行智能负样本选择，提高UMAP质量")
        
        # 返回训练结果
        results = {
            'train_losses': train_losses,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'epochs_trained': epoch + 1 if early_stopped else epochs,
            'early_stopped': early_stopped,
            'best_loss': best_loss
        }
        
        return results

    def predict(self, nadata) -> np.ndarray:
        """
        模型预测（降维）
        
        Args:
            nadata: nadata对象
            
        Returns:
            降维后的嵌入结果
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        self.model.eval()
        with torch.no_grad():
            X = nadata.X
            
            # 数据标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            encoded, _ = self.model(X_tensor)
            return encoded.cpu().numpy()
    
    def evaluate(self, nadata, split='test') -> Dict[str, float]:
        """
        模型评估
        
        Args:
            nadata: nadata对象
            split: 评估的数据集分割
            
        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 获取数据索引
        indices = nadata.Model.get_indices(split)
        if indices is None:
            raise ValueError(f"未找到{split}集的索引")
        
        # 根据索引获取数据
        X = nadata.X[indices]
        
        # 获取嵌入结果
        embeddings = self.predict(nadata)
        embeddings_split = embeddings[indices]
        
        # 计算降维质量指标
        try:
            # 重构误差
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                encoded, decoded = self.model(X_tensor)
                reconstruction_error = F.mse_loss(decoded, X_tensor).item()
            
            # 如果有关联的标签，计算聚类指标
            if hasattr(nadata, 'Meta') and nadata.Meta is not None:
                target_col = self.config.get('dataset', {}).get('target_column', 'target')
                if target_col in nadata.Meta.columns:
                    labels = nadata.Meta.iloc[indices][target_col].values
                    
                    # 计算聚类指标
                    silhouette = silhouette_score(embeddings_split, labels)
                    calinski_harabasz = calinski_harabasz_score(embeddings_split, labels)
                    davies_bouldin = davies_bouldin_score(embeddings_split, labels)
                    
                    results = {
                        'reconstruction_error': reconstruction_error,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski_harabasz,
                        'davies_bouldin_score': davies_bouldin
                    }
                else:
                    results = {
                        'reconstruction_error': reconstruction_error
                    }
            else:
                results = {
                    'reconstruction_error': reconstruction_error
                }
            
        except Exception as e:
            self.logger.error(f"计算评估指标时出现错误: {e}")
            results = {
                'reconstruction_error': float('inf')
            }
        
        # 保存评估结果到Model容器
        eval_results = nadata.Model.get_metadata('evaluation_results') or {}
        eval_results[split] = results
        nadata.Model.add_metadata('evaluation_results', eval_results)
        
        self.logger.info(f"模型评估完成 - {split}集:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def explain(self, nadata, method='importance') -> Dict[str, Any]:
        """
        模型解释
        
        Args:
            nadata: nadata对象
            method: 解释方法
            
        Returns:
            解释结果字典
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        if method == 'importance':
            try:
                # 获取嵌入结果
                embeddings = self.predict(nadata)
                
                # 计算特征重要性（基于重构误差）
                feature_importance = self._calculate_feature_importance(nadata)
                
                # 排序并获取前20个重要特征
                top_indices = np.argsort(feature_importance)[::-1][:20]
                top_features = [nadata.Var.iloc[i]['Gene'] for i in top_indices]
                top_scores = feature_importance[top_indices]
                
                # 打印20个top_features
                self.logger.info(f"  - Top 20 重要基因:")
                self.logger.info(f"    {'排名':<4} {'基因名':<15} {'重要性分数':<12}")
                self.logger.info(f"    {'-'*4} {'-'*15} {'-'*12}")
                for i, (gene, score) in enumerate(zip(top_features, top_scores)):
                    self.logger.info(f"    {i+1:<4} {gene:<15} {score:<12.4f}")
                
                explain_results = {
                    'importance': {
                        'top_features': top_features,
                        'importance_scores': top_scores.tolist(),
                        'embeddings': embeddings.tolist(),
                        'feature_importance': feature_importance.tolist()
                    }
                }
                
                # 保存解释结果
                nadata.uns['nnea_umap_explain'] = explain_results
                
                self.logger.info(f"模型解释完成:")
                return explain_results
                
            except Exception as e:
                self.logger.error(f"模型解释失败: {e}")
                return {}
        else:
            raise ValueError(f"不支持的解释方法: {method}")
    
    def _calculate_feature_importance(self, nadata) -> np.ndarray:
        """
        计算特征重要性
        
        Args:
            nadata: nadata对象
            
        Returns:
            特征重要性数组
        """
        X = nadata.X
        feature_importance = np.zeros(X.shape[1])
        
        # 使用重构误差作为重要性指标
        for i in range(X.shape[1]):
            # 创建扰动数据
            X_perturbed = X.copy()
            X_perturbed[:, i] = 0  # 将第i个特征置零
            
            # 计算重构误差
            X_tensor = torch.FloatTensor(X_perturbed).to(self.device)
            with torch.no_grad():
                encoded, decoded = self.model(X_tensor)
                reconstruction_error = F.mse_loss(decoded, X_tensor).item()
            
            feature_importance[i] = reconstruction_error
        
        return feature_importance
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型状态
        
        Args:
            save_path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未构建")
        
        # 保存模型状态字典
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'umap_loss_state_dict': self.umap_loss.state_dict(),
            'config': self.config,
            'device': self.device,
            'is_trained': self.is_trained
        }, save_path)
        
        self.logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        加载模型状态
        
        Args:
            load_path: 加载路径
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        # 加载模型状态字典
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.umap_loss.load_state_dict(checkpoint['umap_loss_state_dict'])
        
        # 更新其他属性
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        if 'is_trained' in checkpoint:
            self.is_trained = checkpoint['is_trained']
        
        self.logger.info(f"模型已从 {load_path} 加载")
    
    def plot_umap_results(self, nadata, title: str = "NNEA UMAP Visualization", 
                         figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        可视化UMAP结果
        
        Args:
            nadata: nadata对象
            title: 图表标题
            figsize: 图表大小
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 获取嵌入结果
        embeddings = self.predict(nadata)
        
        # 获取标签（如果有）
        labels = None
        if hasattr(nadata, 'Meta') and nadata.Meta is not None:
            target_col = self.config.get('dataset', {}).get('target_column', 'target')
            if target_col in nadata.Meta.columns:
                labels = nadata.Meta[target_col].values
        
        # 创建可视化
        plt.figure(figsize=figsize)
        
        if labels is not None:
            # 有标签的情况
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                           c=[colors[i]], label=f'Class {label}', alpha=0.7)
        else:
            # 无标签的情况
            plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)
        
        plt.title(title)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        
        if labels is not None:
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        self.logger.info(f"UMAP可视化已完成: {title}")

    def get_cache_info(self):
        """
        获取缓存信息
        
        Returns:
            缓存信息字典
        """
        if hasattr(self.umap_loss, 'is_fitted') and self.umap_loss.is_fitted:
            return {
                'pca_fitted': True,
                'nbr_indices_shape': self.umap_loss.nbr_indices.shape if self.umap_loss.nbr_indices is not None else None,
                'pca_components': self.umap_loss.pca.n_components_ if self.umap_loss.pca else 0,
                'original_data_shape': self.umap_loss.original_data.shape if self.umap_loss.original_data is not None else None,
                'global_distances_shape': self.umap_loss.global_distances.shape if self.umap_loss.global_distances is not None else None,
                'smart_negative_sampling': True
            }
        else:
            return {
                'pca_fitted': False,
                'nbr_indices_shape': None,
                'pca_components': 0,
                'original_data_shape': None,
                'global_distances_shape': None,
                'smart_negative_sampling': False
            }
