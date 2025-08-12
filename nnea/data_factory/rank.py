"""
Survival ranking matrix module (na.rank)
Returns rank_exp and sort_exp, refer to utils.io_utils.py
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats


class rank:
    """
    Survival ranking matrix class, provides sorting and ranking functionality
    """
    
    @staticmethod
    def expToRank(nadata, method: str = "rank") -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert expression matrix to ranking matrix
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing expression matrix
        method : str
            Ranking method: 'rank', 'percentile', 'zscore'
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (rank_exp, sort_exp) ranking matrix and sorted matrix
        """
        if nadata.X is None:
            raise ValueError("Expression matrix X is None")
        
        X = nadata.X
        
        if method == "rank":
            # Simple ranking
            rank_exp = np.zeros_like(X)
            sort_exp = np.zeros_like(X)
            
            for i in range(X.shape[0]):
                # Rank each gene
                sorted_indices = np.argsort(X[i, :])
                sort_exp[i, :] = X[i, sorted_indices]
                
                # Calculate ranking
                rank_exp[i, sorted_indices] = np.arange(X.shape[1])
                
        elif method == "percentile":
            # Percentile ranking
            rank_exp = np.zeros_like(X)
            sort_exp = np.zeros_like(X)
            
            for i in range(X.shape[0]):
                # Calculate percentiles
                percentiles = stats.rankdata(X[i, :], method='average') / len(X[i, :]) * 100
                rank_exp[i, :] = percentiles
                
                # Sorted expression values
                sorted_indices = np.argsort(X[i, :])
                sort_exp[i, :] = X[i, sorted_indices]
                
        elif method == "zscore":
            # Z-score standardization
            rank_exp = np.zeros_like(X)
            sort_exp = np.zeros_like(X)
            
            for i in range(X.shape[0]):
                # Calculate Z-score
                mean_val = np.mean(X[i, :])
                std_val = np.std(X[i, :])
                if std_val > 0:
                    z_scores = (X[i, :] - mean_val) / std_val
                else:
                    z_scores = np.zeros_like(X[i, :])
                
                rank_exp[i, :] = z_scores
                
                # Sorted expression values
                sorted_indices = np.argsort(X[i, :])
                sort_exp[i, :] = X[i, sorted_indices]
                
        else:
            raise ValueError(f"Unsupported ranking method: {method}")
        
        return rank_exp, sort_exp
    
    @staticmethod
    def expToRank_sparse(nadata, method: str = "rank") -> Tuple[np.ndarray, np.ndarray]:
        """
        Sparse matrix version of ranking conversion
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing expression matrix
        method : str
            Ranking method: 'rank', 'percentile', 'zscore'
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (rank_exp, sort_exp) ranking matrix and sorted matrix
        """
        if nadata.X is None:
            raise ValueError("Expression matrix X is None")
        
        # Convert to dense matrix
        if hasattr(nadata.X, 'toarray'):
            X = nadata.X.toarray()
        else:
            X = nadata.X
        
        return rank.expToRank(nadata, method)
    
    @staticmethod
    def survival_rank(nadata, time_col: str = "time", event_col: str = "event") -> Tuple[np.ndarray, np.ndarray]:
        """
        Survival analysis ranking
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        time_col : str
            Time column name
        event_col : str
            Event column name
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (rank_exp, sort_exp) ranking matrix and sorted matrix
        """
        if nadata.X is None or nadata.Meta is None:
            raise ValueError("Expression matrix X or Meta data is None")
        
        if time_col not in nadata.Meta.columns or event_col not in nadata.Meta.columns:
            raise ValueError(f"Time column '{time_col}' or event column '{event_col}' not found")
        
        X = nadata.X
        times = nadata.Meta[time_col].values
        events = nadata.Meta[event_col].values
        
        # Calculate survival ranking
        rank_exp = np.zeros_like(X)
        sort_exp = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            # Calculate survival ranking for each gene
            gene_exp = X[i, :]
            
            # Sort by expression values
            sorted_indices = np.argsort(gene_exp)
            sort_exp[i, :] = gene_exp[sorted_indices]
            
            # Calculate survival ranking
            survival_ranks = _calculate_survival_ranks(times, events, sorted_indices)
            rank_exp[i, sorted_indices] = survival_ranks
        
        return rank_exp, sort_exp
    
    @staticmethod
    def correlation_rank(nadata, target_col: str = "target", method: str = "pearson") -> Tuple[np.ndarray, np.ndarray]:
        """
        Correlation-based ranking
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        target_col : str
            Target variable column name
        method : str
            Correlation method: 'pearson', 'spearman', 'kendall'
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (rank_exp, sort_exp) ranking matrix and sorted matrix
        """
        if nadata.X is None or nadata.Meta is None:
            raise ValueError("Expression matrix X or Meta data is None")
        
        if target_col not in nadata.Meta.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = nadata.X
        target = nadata.Meta[target_col].values
        
        # 计算相关性
        correlations = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if method == "pearson":
                corr, _ = stats.pearsonr(X[i, :], target)
            elif method == "spearman":
                corr, _ = stats.spearmanr(X[i, :], target)
            elif method == "kendall":
                corr, _ = stats.kendalltau(X[i, :], target)
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
            
            correlations[i] = corr if not np.isnan(corr) else 0
        
        # 按相关性排序
        sorted_indices = np.argsort(np.abs(correlations))[::-1]
        
        rank_exp = np.zeros_like(X)
        sort_exp = np.zeros_like(X)
        
        for i, gene_idx in enumerate(sorted_indices):
            rank_exp[gene_idx, :] = i
            sort_exp[i, :] = X[gene_idx, :]
        
        return rank_exp, sort_exp
    
    @staticmethod
    def differential_expression_rank(nadata, group_col: str = "group", method: str = "t_test") -> Tuple[np.ndarray, np.ndarray]:
        """
        差异表达排名
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        group_col : str
            分组列名
        method : str
            统计方法：'t_test', 'wilcoxon', 'mann_whitney'
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (rank_exp, sort_exp) 排名矩阵和排序矩阵
        """
        if nadata.X is None or nadata.Meta is None:
            raise ValueError("Expression matrix X or Meta data is None")
        
        if group_col not in nadata.Meta.columns:
            raise ValueError(f"Group column '{group_col}' not found")
        
        X = nadata.X
        groups = nadata.Meta[group_col].values
        
        # 获取唯一分组
        unique_groups = np.unique(groups)
        if len(unique_groups) != 2:
            raise ValueError("Differential expression analysis requires exactly 2 groups")
        
        group1, group2 = unique_groups
        group1_mask = groups == group1
        group2_mask = groups == group2
        
        # 计算统计量
        statistics = np.zeros(X.shape[0])
        p_values = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            gene_exp = X[i, :]
            group1_exp = gene_exp[group1_mask]
            group2_exp = gene_exp[group2_mask]
            
            if method == "t_test":
                stat, p_val = stats.ttest_ind(group1_exp, group2_exp)
            elif method == "wilcoxon":
                stat, p_val = stats.wilcoxon(group1_exp, group2_exp)
            elif method == "mann_whitney":
                stat, p_val = stats.mannwhitneyu(group1_exp, group2_exp, alternative='two-sided')
            else:
                raise ValueError(f"Unsupported statistical method: {method}")
            
            statistics[i] = stat if not np.isnan(stat) else 0
            p_values[i] = p_val if not np.isnan(p_val) else 1
        
        # 按p值排序
        sorted_indices = np.argsort(p_values)
        
        rank_exp = np.zeros_like(X)
        sort_exp = np.zeros_like(X)
        
        for i, gene_idx in enumerate(sorted_indices):
            rank_exp[gene_idx, :] = i
            sort_exp[i, :] = X[gene_idx, :]
        
        return rank_exp, sort_exp


def _calculate_survival_ranks(times: np.ndarray, events: np.ndarray, sorted_indices: np.ndarray) -> np.ndarray:
    """
    计算生存排名
    
    Parameters:
    -----------
    times : np.ndarray
        生存时间
    events : np.ndarray
        事件指示器
    sorted_indices : np.ndarray
        排序索引
        
    Returns:
    --------
    np.ndarray
        生存排名
    """
    n_samples = len(times)
    ranks = np.zeros(n_samples)
    
    # 按时间排序
    time_sorted_indices = np.argsort(times)
    
    for i, idx in enumerate(time_sorted_indices):
        if events[idx]:
            # 事件样本
            ranks[idx] = i + 1
        else:
            # 删失样本
            ranks[idx] = i + 0.5
    
    return ranks 