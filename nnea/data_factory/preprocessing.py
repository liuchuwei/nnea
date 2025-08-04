"""
数据预处理模块（na.pp）
包含数据标准化、缺失值处理、异常值检测和处理、基因/样本过滤等功能
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import warnings


class pp:
    """
    数据预处理类，提供各种预处理方法
    """
    
    @staticmethod
    def fillna(X, method: str = "mean", fill_value: float = 0):
        """
        处理缺失值
        
        Parameters:
        -----------
        X : np.ndarray
            输入数据矩阵
        method : str
            处理方法：'mean', 'median', 'zero', 'drop'
        fill_value : float
            填充值，用于'zero'方法
            
        Returns:
        --------
        np.ndarray
            处理后的数据矩阵
        """
        if X is None:
            return X
        
        if not np.isnan(X).any():
            return X
        
        if method == "mean":
            # 检查每一列是否全为NaN，若是则用0填充，否则用均值填充
            nan_all_col = np.isnan(X).all(axis=0)
            if nan_all_col.any():
                X[:, nan_all_col] = 0
            # 对剩余有NaN的列用均值填充
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            
        elif method == "median":
            # 检查每一列是否全为NaN，若是则用0填充，否则用中位数填充
            nan_all_col = np.isnan(X).all(axis=0)
            if nan_all_col.any():
                X[:, nan_all_col] = 0
            # 对剩余有NaN的列用中位数填充
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            
        elif method == "zero":
            # 用指定值填充
            X = np.nan_to_num(X, nan=fill_value)
            
        elif method == "drop":
            # 删除包含缺失值的行
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            
        else:
            raise ValueError(f"Unsupported fillna method: {method}")
        
        return X
    
    @staticmethod
    def scale(X, method: str = "standard"):
        """
        数据标准化
        
        Parameters:
        -----------
        X : np.ndarray
            输入数据矩阵
        method : str
            标准化方法：'standard', 'minmax', 'robust'
            
        Returns:
        --------
        np.ndarray
            标准化后的数据矩阵
        """
        if X is None:
            return X
        
        if method == "standard":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
        elif method == "minmax":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
        elif method == "robust":
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
        else:
            raise ValueError(f"Unsupported scale method: {method}")
        
        return X_scaled
    
    @staticmethod
    def x_train_test(X, nadata, test_size: float = 0.2, random_state: int = 42):
        """
        获取训练集和测试集的X数据
        
        Parameters:
        -----------
        X : np.ndarray
            特征数据矩阵
        nadata : nadata对象
            包含划分信息的nadata对象
        test_size : float
            测试集比例
        random_state : int
            随机种子
            
        Returns:
        --------
        tuple
            (X_train, X_test)
        """
        if hasattr(nadata, 'Model') and hasattr(nadata.Model, 'indices'):
            # 使用已保存的划分信息
            train_idx = nadata.Model.indices['train']
            test_idx = nadata.Model.indices['test']
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            # 使用随机划分
            from sklearn.model_selection import train_test_split
            indices = np.arange(X.shape[0])
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )
            X_train, X_test = X[train_idx], X[test_idx]
        
        return X_train, X_test
    
    @staticmethod
    def y_train_test(y, nadata, test_size: float = 0.2, random_state: int = 42):
        """
        获取训练集和测试集的y数据
        
        Parameters:
        -----------
        y : pd.Series or np.ndarray
            标签数据
        nadata : nadata对象
            包含划分信息的nadata对象
        test_size : float
            测试集比例
        random_state : int
            随机种子
            
        Returns:
        --------
        tuple
            (y_train, y_test)
        """
        if hasattr(nadata, 'Model') and hasattr(nadata.Model, 'indices'):
            # 使用已保存的划分信息
            train_idx = nadata.Model.indices['train']
            test_idx = nadata.Model.indices['test']
            if isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]
        else:
            # 使用随机划分
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(y))
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )
            if isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]
        
        return y_train, y_test

    @staticmethod
    def normalize(nadata, method: str = "zscore", scale_factor: float = 10000):
        """
        数据标准化
        
        Parameters:
        -----------
        nadata : nadata对象
            包含表达矩阵的nadata对象
        method : str
            标准化方法：'zscore', 'minmax', 'robust', 'quantile', 'cell_by_gene'
        scale_factor : float
            缩放因子，用于cell_by_gene方法
            
        Returns:
        --------
        nadata
            标准化后的nadata对象
        """
        if nadata.X is None:
            raise ValueError("Expression matrix X is None")
        
        X = nadata.X.copy()
        
        if method == "zscore":
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X.T).T
            
        elif method == "minmax":
            scaler = MinMaxScaler()
            X_normalized = scaler.fit_transform(X.T).T
            
        elif method == "robust":
            scaler = RobustScaler()
            X_normalized = scaler.fit_transform(X.T).T
            
        elif method == "quantile":
            # 分位数标准化
            X_normalized = np.zeros_like(X)
            for i in range(X.shape[0]):
                gene_exp = X[i, :]
                q75, q25 = np.percentile(gene_exp, [75, 25])
                if q75 != q25:
                    X_normalized[i, :] = (gene_exp - q25) / (q75 - q25)
                else:
                    X_normalized[i, :] = gene_exp
                    
        elif method == "cell_by_gene":
            # 按细胞标准化（单细胞数据常用）
            X_normalized = _normalize_cell_by_gene(X, scale_factor)
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        nadata.X = X_normalized
        return nadata
    
    @staticmethod
    def handle_missing_values(nadata, method: str = "drop", fill_value: float = 0):
        """
        缺失值处理
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            处理方法：'drop', 'fill', 'interpolate'
        fill_value : float
            填充值，用于fill方法
            
        Returns:
        --------
        nadata
            处理后的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X
        
        if method == "drop":
            # 删除包含缺失值的基因或样本
            # 删除基因（行）
            gene_mask = ~np.isnan(X).any(axis=1)
            X_clean = X[gene_mask, :]
            if nadata.Var is not None:
                nadata.Var = nadata.Var.iloc[gene_mask]
            
            # 删除样本（列）
            sample_mask = ~np.isnan(X_clean).any(axis=0)
            X_clean = X_clean[:, sample_mask]
            if nadata.Meta is not None:
                nadata.Meta = nadata.Meta.iloc[sample_mask]
            
            nadata.X = X_clean
            
        elif method == "fill":
            # 用指定值填充
            X_filled = np.nan_to_num(X, nan=fill_value)
            nadata.X = X_filled
            
        elif method == "interpolate":
            # 插值填充
            from scipy.interpolate import interp1d
            
            X_interpolated = X.copy()
            for i in range(X.shape[0]):
                gene_exp = X[i, :]
                if np.isnan(gene_exp).any():
                    # 找到非缺失值的索引
                    valid_idx = ~np.isnan(gene_exp)
                    if valid_idx.sum() > 1:
                        # 插值
                        f = interp1d(np.where(valid_idx)[0], gene_exp[valid_idx], 
                                    kind='linear', fill_value='extrapolate')
                        all_idx = np.arange(len(gene_exp))
                        X_interpolated[i, :] = f(all_idx)
                    else:
                        # 如果只有一个有效值，用该值填充
                        valid_value = gene_exp[valid_idx][0]
                        X_interpolated[i, :] = valid_value
            
            nadata.X = X_interpolated
            
        else:
            raise ValueError(f"Unsupported missing value method: {method}")
        
        return nadata
    
    @staticmethod
    def detect_outliers(nadata, method: str = "iqr", threshold: float = 1.5):
        """
        异常值检测
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            检测方法：'iqr', 'zscore', 'isolation_forest'
        threshold : float
            阈值
            
        Returns:
        --------
        nadata
            处理后的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X
        outlier_mask = np.zeros(X.shape, dtype=bool)
        
        if method == "iqr":
            # IQR方法
            for i in range(X.shape[0]):
                gene_exp = X[i, :]
                Q1 = np.percentile(gene_exp, 25)
                Q3 = np.percentile(gene_exp, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask[i, :] = (gene_exp < lower_bound) | (gene_exp > upper_bound)
                
        elif method == "zscore":
            # Z-score方法
            for i in range(X.shape[0]):
                gene_exp = X[i, :]
                z_scores = np.abs((gene_exp - np.mean(gene_exp)) / np.std(gene_exp))
                outlier_mask[i, :] = z_scores > threshold
                
        elif method == "isolation_forest":
            # 隔离森林方法
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X.T)
            outlier_mask = (outlier_labels == -1).reshape(X.shape[1], X.shape[0]).T
            
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        # 将异常值设为NaN
        X_clean = X.copy()
        X_clean[outlier_mask] = np.nan
        nadata.X = X_clean
        
        return nadata
    
    @staticmethod
    def filter_genes(nadata, method: str = "variance", **kwargs):
        """
        基因过滤
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            过滤方法：'variance', 'top_k', 'expression_threshold'
        **kwargs : 
            其他参数
            
        Returns:
        --------
        nadata
            过滤后的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X
        
        if method == "variance":
            # 方差过滤
            threshold = kwargs.get('threshold', 0.01)
            selector = VarianceThreshold(threshold=threshold)
            X_filtered = selector.fit_transform(X.T).T
            gene_mask = selector.get_support()
            
        elif method == "top_k":
            # 选择前k个基因
            k = kwargs.get('k', 1000)
            if k >= X.shape[0]:
                return nadata
            
            # 计算方差
            variances = np.var(X, axis=1)
            top_indices = np.argsort(variances)[-k:]
            X_filtered = X[top_indices, :]
            gene_mask = np.zeros(X.shape[0], dtype=bool)
            gene_mask[top_indices] = True
            
        elif method == "expression_threshold":
            # 表达量阈值过滤
            threshold = kwargs.get('threshold', 0)
            min_cells = kwargs.get('min_cells', 1)
            
            # 计算每个基因在多少个细胞中表达
            expressed_cells = (X > threshold).sum(axis=1)
            gene_mask = expressed_cells >= min_cells
            X_filtered = X[gene_mask, :]
            
        else:
            raise ValueError(f"Unsupported gene filtering method: {method}")
        
        nadata.X = X_filtered
        if nadata.Var is not None:
            nadata.Var = nadata.Var.iloc[gene_mask]
        if nadata.Prior is not None:
            nadata.Prior = nadata.Prior[:, gene_mask]
        
        return nadata
    
    @staticmethod
    def filter_samples(nadata, method: str = "quality", **kwargs):
        """
        样本过滤
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            过滤方法：'quality', 'expression_threshold'
        **kwargs : 
            其他参数
            
        Returns:
        --------
        nadata
            过滤后的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X
        
        if method == "quality":
            # 质量过滤
            min_genes = kwargs.get('min_genes', 1)
            max_genes = kwargs.get('max_genes', float('inf'))
            
            # 计算每个样本表达的基因数
            expressed_genes = (X > 0).sum(axis=0)
            sample_mask = (expressed_genes >= min_genes) & (expressed_genes <= max_genes)
            
        elif method == "expression_threshold":
            # 表达量阈值过滤
            threshold = kwargs.get('threshold', 0)
            min_genes = kwargs.get('min_genes', 1)
            
            # 计算每个样本表达量超过阈值的基因数
            expressed_genes = (X > threshold).sum(axis=0)
            sample_mask = expressed_genes >= min_genes
            
        else:
            raise ValueError(f"Unsupported sample filtering method: {method}")
        
        nadata.X = X[:, sample_mask]
        if nadata.Meta is not None:
            nadata.Meta = nadata.Meta.iloc[sample_mask]
        
        return nadata
    
    @staticmethod
    def split_data(nadata, test_size: float = 0.2,
                   random_state: int = 42, strategy: str = "random"):
        """
        数据划分
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        test_size : float
            测试集比例
        val_size : float
            验证集比例
        random_state : int
            随机种子
        strategy : str
            划分策略：'random', 'stratified'
            
        Returns:
        --------
        nadata
            包含划分信息的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        n_samples = nadata.X.shape[0]
        indices = np.arange(n_samples)
        
        if strategy == "random":
            from sklearn.model_selection import train_test_split
            
            # 首先划分出测试集
            train_indices, test_indices = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )

            
        elif strategy == "stratified":
            from sklearn.model_selection import train_test_split
            
            # 需要目标变量进行分层抽样
            if nadata.Meta is not None and 'target' in nadata.Meta.columns:
                target = nadata.Meta['target']
                
                # 首先划分出测试集
                train_indices, test_indices = train_test_split(
                    indices, test_size=test_size, random_state=random_state, 
                    stratify=target
                )

            else:
                warnings.warn("No target column found for stratified sampling, using random split")
                return pp.split_data(nadata, test_size, random_state, "random")
        
        # 保存划分信息到nadata.Model.indices中
        nadata.Model.set_indices(
            train_idx=train_indices,
            test_idx=test_indices,
        )
        
        # 同时保存策略信息到config中
        if not hasattr(nadata, 'config'):
            nadata.config = {}
        nadata.config['data_split_strategy'] = strategy
        
        return nadata


def _normalize_cell_by_gene(X: np.ndarray, scale_factor: float = 10000) -> np.ndarray:
    """
    按细胞标准化（单细胞数据常用）
    
    Parameters:
    -----------
    X : np.ndarray
        表达矩阵
    scale_factor : float
        缩放因子
        
    Returns:
    --------
    np.ndarray
        标准化后的表达矩阵
    """
    # 计算每个细胞的总表达量
    cell_sums = np.sum(X, axis=0)
    
    # 标准化
    X_normalized = X / cell_sums * scale_factor
    
    # 对数转换
    X_normalized = np.log1p(X_normalized)
    
    return X_normalized 