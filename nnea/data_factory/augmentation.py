"""
数据增强模块（na.au）
包含数据扰动、噪声添加、数据平衡等功能
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings


class au:
    """
    数据增强类，提供各种数据增强方法
    """
    
    @staticmethod
    def add_noise(nadata, method: str = "gaussian", **kwargs):
        """
        添加噪声
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            噪声类型：'gaussian', 'poisson', 'dropout'
        **kwargs : 
            其他参数
            
        Returns:
        --------
        nadata
            添加噪声后的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X.copy()
        
        if method == "gaussian":
            # 高斯噪声
            std = kwargs.get('std', 0.1)
            noise = np.random.normal(0, std, X.shape)
            X_noisy = X + noise
            
        elif method == "poisson":
            # 泊松噪声
            intensity = kwargs.get('intensity', 0.1)
            noise = np.random.poisson(intensity, X.shape)
            X_noisy = X + noise
            
        elif method == "dropout":
            # Dropout噪声
            rate = kwargs.get('rate', 0.1)
            mask = np.random.binomial(1, 1-rate, X.shape)
            X_noisy = X * mask
            
        else:
            raise ValueError(f"Unsupported noise method: {method}")
        
        nadata.X = X_noisy
        return nadata
    
    @staticmethod
    def balance_data(nadata, method: str = "smote", target_col: str = "target", **kwargs):
        """
        数据平衡
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            平衡方法：'smote', 'adasyn', 'borderline_smote', 'undersample', 'combine'
        target_col : str
            目标变量列名
        **kwargs : 
            其他参数
            
        Returns:
        --------
        nadata
            平衡后的nadata对象
        """
        if nadata.X is None or nadata.Meta is None:
            return nadata
        
        if target_col not in nadata.Meta.columns:
            raise ValueError(f"Target column '{target_col}' not found in Meta data")
        
        X = nadata.X.T  # 转置为(样本数, 特征数)
        y = nadata.Meta[target_col].values
        
        if method == "smote":
            # SMOTE过采样
            k_neighbors = kwargs.get('k_neighbors', 5)
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)
            
        elif method == "adasyn":
            # ADASYN过采样
            k_neighbors = kwargs.get('k_neighbors', 5)
            sampler = ADASYN(k_neighbors=k_neighbors, random_state=42)
            
        elif method == "borderline_smote":
            # Borderline SMOTE
            k_neighbors = kwargs.get('k_neighbors', 5)
            sampler = BorderlineSMOTE(k_neighbors=k_neighbors, random_state=42)
            
        elif method == "undersample":
            # 欠采样
            sampler = RandomUnderSampler(random_state=42)
            
        elif method == "combine":
            # 组合方法
            if kwargs.get('method', 'smoteenn') == 'smoteenn':
                sampler = SMOTEENN(random_state=42)
            else:
                sampler = SMOTETomek(random_state=42)
                
        else:
            raise ValueError(f"Unsupported balancing method: {method}")
        
        # 执行采样
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # 更新数据
        nadata.X = X_resampled.T
        nadata.Meta = nadata.Meta.iloc[:len(y_resampled)].copy()
        nadata.Meta[target_col] = y_resampled
        
        return nadata
    
    @staticmethod
    def perturb_data(nadata, method: str = "random", **kwargs):
        """
        数据扰动
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            扰动方法：'random', 'systematic', 'feature_wise'
        **kwargs : 
            其他参数
            
        Returns:
        --------
        nadata
            扰动后的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X.copy()
        
        if method == "random":
            # 随机扰动
            scale = kwargs.get('scale', 0.1)
            perturbation = np.random.uniform(-scale, scale, X.shape)
            X_perturbed = X + perturbation
            
        elif method == "systematic":
            # 系统性扰动
            bias = kwargs.get('bias', 0.05)
            X_perturbed = X + bias
            
        elif method == "feature_wise":
            # 特征级扰动
            scale = kwargs.get('scale', 0.1)
            perturbation = np.random.normal(0, scale, X.shape)
            # 对每个特征应用不同的扰动
            for i in range(X.shape[0]):
                feature_scale = np.random.uniform(0.5, 1.5) * scale
                perturbation[i, :] *= feature_scale
            X_perturbed = X + perturbation
            
        else:
            raise ValueError(f"Unsupported perturbation method: {method}")
        
        nadata.X = X_perturbed
        return nadata
    
    @staticmethod
    def augment_single_cell(nadata, method: str = "synthetic", **kwargs):
        """
        单细胞数据增强
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            增强方法：'synthetic', 'mixup', 'cutmix'
        **kwargs : 
            其他参数
            
        Returns:
        --------
        nadata
            增强后的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X.copy()
        n_genes, n_cells = X.shape
        
        if method == "synthetic":
            # 合成细胞生成
            n_synthetic = kwargs.get('n_synthetic', n_cells // 2)
            synthetic_cells = np.zeros((n_genes, n_synthetic))
            
            for i in range(n_synthetic):
                # 随机选择两个细胞进行混合
                cell1, cell2 = np.random.choice(n_cells, 2, replace=False)
                alpha = np.random.uniform(0.3, 0.7)
                synthetic_cells[:, i] = alpha * X[:, cell1] + (1 - alpha) * X[:, cell2]
            
            # 合并原始数据和合成数据
            X_augmented = np.concatenate([X, synthetic_cells], axis=1)
            
            # 更新Meta数据
            if nadata.Meta is not None:
                original_meta = nadata.Meta.copy()
                synthetic_meta = original_meta.iloc[:n_synthetic].copy()
                synthetic_meta.index = range(len(original_meta), len(original_meta) + n_synthetic)
                nadata.Meta = pd.concat([original_meta, synthetic_meta], ignore_index=True)
            
        elif method == "mixup":
            # Mixup增强
            alpha = kwargs.get('alpha', 0.2)
            n_augmented = kwargs.get('n_augmented', n_cells)
            X_augmented = np.zeros((n_genes, n_augmented))
            
            for i in range(n_augmented):
                # 随机选择两个细胞
                cell1, cell2 = np.random.choice(n_cells, 2, replace=False)
                # 生成混合权重
                lam = np.random.beta(alpha, alpha)
                X_augmented[:, i] = lam * X[:, cell1] + (1 - lam) * X[:, cell2]
            
            # 更新Meta数据
            if nadata.Meta is not None:
                original_meta = nadata.Meta.copy()
                augmented_meta = original_meta.iloc[:n_augmented].copy()
                augmented_meta.index = range(len(original_meta), len(original_meta) + n_augmented)
                nadata.Meta = pd.concat([original_meta, augmented_meta], ignore_index=True)
                
        elif method == "cutmix":
            # CutMix增强
            n_augmented = kwargs.get('n_augmented', n_cells)
            X_augmented = np.zeros((n_genes, n_augmented))
            
            for i in range(n_augmented):
                # 随机选择两个细胞
                cell1, cell2 = np.random.choice(n_cells, 2, replace=False)
                # 随机选择基因子集
                n_cut = np.random.randint(1, n_genes // 2)
                cut_indices = np.random.choice(n_genes, n_cut, replace=False)
                
                # 混合基因表达
                X_augmented[:, i] = X[:, cell1].copy()
                X_augmented[cut_indices, i] = X[cut_indices, cell2]
            
            # 更新Meta数据
            if nadata.Meta is not None:
                original_meta = nadata.Meta.copy()
                augmented_meta = original_meta.iloc[:n_augmented].copy()
                augmented_meta.index = range(len(original_meta), len(original_meta) + n_augmented)
                nadata.Meta = pd.concat([original_meta, augmented_meta], ignore_index=True)
                
        else:
            raise ValueError(f"Unsupported single-cell augmentation method: {method}")
        
        nadata.X = X_augmented
        return nadata
    
    @staticmethod
    def time_series_augmentation(nadata, method: str = "temporal", **kwargs):
        """
        时间序列数据增强
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            增强方法：'temporal', 'frequency', 'noise_injection'
        **kwargs : 
            其他参数
            
        Returns:
        --------
        nadata
            增强后的nadata对象
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X.copy()
        
        if method == "temporal":
            # 时间窗口滑动
            window_size = kwargs.get('window_size', 3)
            stride = kwargs.get('stride', 1)
            
            augmented_data = []
            for i in range(0, X.shape[1] - window_size + 1, stride):
                window_data = X[:, i:i+window_size]
                # 计算窗口内的统计特征
                mean_data = np.mean(window_data, axis=1, keepdims=True)
                augmented_data.append(mean_data)
            
            if augmented_data:
                X_augmented = np.concatenate(augmented_data, axis=1)
                nadata.X = X_augmented
                
        elif method == "frequency":
            # 频域增强
            from scipy.fft import fft, ifft
            
            # 对每个基因进行FFT
            X_fft = fft(X, axis=1)
            
            # 添加频域噪声
            noise_scale = kwargs.get('noise_scale', 0.1)
            noise = np.random.normal(0, noise_scale, X_fft.shape)
            X_fft_noisy = X_fft + noise
            
            # 逆FFT
            X_augmented = np.real(ifft(X_fft_noisy, axis=1))
            nadata.X = X_augmented
            
        elif method == "noise_injection":
            # 噪声注入
            noise_type = kwargs.get('noise_type', 'gaussian')
            noise_scale = kwargs.get('noise_scale', 0.1)
            
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_scale, X.shape)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-noise_scale, noise_scale, X.shape)
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            
            X_augmented = X + noise
            nadata.X = X_augmented
            
        else:
            raise ValueError(f"Unsupported time series augmentation method: {method}")
        
        return nadata 