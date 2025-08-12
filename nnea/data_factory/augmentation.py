"""
Data Augmentation Module (na.au)
Contains data perturbation, noise addition, data balancing and other functions
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
import warnings


class au:
    """
    Data Augmentation Class, provides various data augmentation methods
    """
    
    @staticmethod
    def add_noise(nadata, method: str = "gaussian", **kwargs):
        """
        Add Noise
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Noise type: 'gaussian', 'poisson', 'dropout'
        **kwargs : 
            Other parameters
            
        Returns:
        --------
        nadata
            nadata object with added noise
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X.copy()
        
        if method == "gaussian":
            # Gaussian noise
            std = kwargs.get('std', 0.1)
            noise = np.random.normal(0, std, X.shape)
            X_noisy = X + noise
            
        elif method == "poisson":
            # Poisson noise
            intensity = kwargs.get('intensity', 0.1)
            noise = np.random.poisson(intensity, X.shape)
            X_noisy = X + noise
            
        elif method == "dropout":
            # Dropout noise
            rate = kwargs.get('rate', 0.1)
            mask = np.random.binomial(1, 1-rate, X.shape)
            X_noisy = X * mask
            
        else:
            raise ValueError(f"Unsupported noise method: {method}")
        
        nadata.X = X_noisy
        return nadata
    
    @staticmethod
    def balance_data(nadata, method: str = "random_oversample", target_col: str = "target", **kwargs):
        """
        Data Balancing
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Balancing method: 'random_oversample', 'random_undersample'
        target_col : str
            Target variable column name
        **kwargs : 
            Other parameters
            
        Returns:
        --------
        nadata
            Balanced nadata object
        """
        if nadata.X is None or nadata.Meta is None:
            return nadata
        
        if target_col not in nadata.Meta.columns:
            raise ValueError(f"Target column '{target_col}' not found in Meta data")
        
        X = nadata.X.T  # Transpose to (samples, features)
        y = nadata.Meta[target_col].values
        
        # Calculate class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_classes = unique_classes[unique_classes != majority_class]
        
        if method == "random_oversample":
            # Simple random oversampling
            max_count = np.max(class_counts)
            X_resampled_list = []
            y_resampled_list = []
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                if len(class_indices) < max_count:
                    # Oversample minority classes
                    oversampled_indices = np.random.choice(
                        class_indices, 
                        size=max_count, 
                        replace=True
                    )
                else:
                    oversampled_indices = class_indices
                
                X_resampled_list.append(X[oversampled_indices])
                y_resampled_list.append(y[oversampled_indices])
            
            X_resampled = np.vstack(X_resampled_list)
            y_resampled = np.concatenate(y_resampled_list)
            
        elif method == "random_undersample":
            # Simple random undersampling
            min_count = np.min(class_counts)
            X_resampled_list = []
            y_resampled_list = []
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                # Undersample majority classes
                undersampled_indices = np.random.choice(
                    class_indices, 
                    size=min_count, 
                    replace=False
                )
                
                X_resampled_list.append(X[undersampled_indices])
                y_resampled_list.append(y[undersampled_indices])
            
            X_resampled = np.vstack(X_resampled_list)
            y_resampled = np.concatenate(y_resampled_list)
                
        else:
            raise ValueError(f"Unsupported balancing method: {method}. Supported methods: 'random_oversample', 'random_undersample'")
        
        # Update data
        nadata.X = X_resampled.T
        nadata.Meta = nadata.Meta.iloc[:len(y_resampled)].copy()
        nadata.Meta[target_col] = y_resampled
        
        return nadata
    
    @staticmethod
    def perturb_data(nadata, method: str = "random", **kwargs):
        """
        Data Perturbation
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Perturbation method: 'random', 'systematic', 'feature_wise'
        **kwargs : 
            Other parameters
            
        Returns:
        --------
        nadata
            Perturbed nadata object
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X.copy()
        
        if method == "random":
            # Random perturbation
            scale = kwargs.get('scale', 0.1)
            perturbation = np.random.uniform(-scale, scale, X.shape)
            X_perturbed = X + perturbation
            
        elif method == "systematic":
            # Systematic perturbation
            bias = kwargs.get('bias', 0.05)
            X_perturbed = X + bias
            
        elif method == "feature_wise":
            # Feature-wise perturbation
            scale = kwargs.get('scale', 0.1)
            perturbation = np.random.normal(0, scale, X.shape)
            # Apply different perturbation to each feature
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
        Single-cell Data Augmentation
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Augmentation method: 'synthetic', 'mixup', 'cutmix'
        **kwargs : 
            Other parameters
            
        Returns:
        --------
        nadata
            Augmented nadata object
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X.copy()
        n_genes, n_cells = X.shape
        
        if method == "synthetic":
            # Synthetic cell generation
            n_synthetic = kwargs.get('n_synthetic', n_cells // 2)
            synthetic_cells = np.zeros((n_genes, n_synthetic))
            
            for i in range(n_synthetic):
                # Randomly select two cells for mixing
                cell1, cell2 = np.random.choice(n_cells, 2, replace=False)
                alpha = np.random.uniform(0.3, 0.7)
                synthetic_cells[:, i] = alpha * X[:, cell1] + (1 - alpha) * X[:, cell2]
            
            # Combine original data and synthetic data
            X_augmented = np.concatenate([X, synthetic_cells], axis=1)
            
            # Update Meta data
            if nadata.Meta is not None:
                original_meta = nadata.Meta.copy()
                synthetic_meta = original_meta.iloc[:n_synthetic].copy()
                synthetic_meta.index = range(len(original_meta), len(original_meta) + n_synthetic)
                nadata.Meta = pd.concat([original_meta, synthetic_meta], ignore_index=True)
            
        elif method == "mixup":
            # Mixup augmentation
            alpha = kwargs.get('alpha', 0.2)
            n_augmented = kwargs.get('n_augmented', n_cells)
            X_augmented = np.zeros((n_genes, n_augmented))
            
            for i in range(n_augmented):
                # Randomly select two cells
                cell1, cell2 = np.random.choice(n_cells, 2, replace=False)
                # Generate mixing weights
                lam = np.random.beta(alpha, alpha)
                X_augmented[:, i] = lam * X[:, cell1] + (1 - lam) * X[:, cell2]
            
            # Update Meta data
            if nadata.Meta is not None:
                original_meta = nadata.Meta.copy()
                augmented_meta = original_meta.iloc[:n_augmented].copy()
                augmented_meta.index = range(len(original_meta), len(original_meta) + n_augmented)
                nadata.Meta = pd.concat([original_meta, augmented_meta], ignore_index=True)
                
        elif method == "cutmix":
            # CutMix augmentation
            n_augmented = kwargs.get('n_augmented', n_cells)
            X_augmented = np.zeros((n_genes, n_augmented))
            
            for i in range(n_augmented):
                # Randomly select two cells
                cell1, cell2 = np.random.choice(n_cells, 2, replace=False)
                # Randomly select gene subset
                n_cut = np.random.randint(1, n_genes // 2)
                cut_indices = np.random.choice(n_genes, n_cut, replace=False)
                
                # Mix gene expression
                X_augmented[:, i] = X[:, cell1].copy()
                X_augmented[cut_indices, i] = X[cut_indices, cell2]
            
            # Update Meta data
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
        Time Series Data Augmentation
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Augmentation method: 'temporal', 'frequency', 'noise_injection'
        **kwargs : 
            Other parameters
            
        Returns:
        --------
        nadata
            Augmented nadata object
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X.copy()
        
        if method == "temporal":
            # Temporal window sliding
            window_size = kwargs.get('window_size', 3)
            stride = kwargs.get('stride', 1)
            
            augmented_data = []
            for i in range(0, X.shape[1] - window_size + 1, stride):
                window_data = X[:, i:i+window_size]
                # Calculate statistical features within the window
                mean_data = np.mean(window_data, axis=1, keepdims=True)
                augmented_data.append(mean_data)
            
            if augmented_data:
                X_augmented = np.concatenate(augmented_data, axis=1)
                nadata.X = X_augmented
                
        elif method == "frequency":
            # Frequency domain augmentation
            from scipy.fft import fft, ifft
            
            # Perform FFT on each gene
            X_fft = fft(X, axis=1)
            
            # Add frequency domain noise
            noise_scale = kwargs.get('noise_scale', 0.1)
            noise = np.random.normal(0, noise_scale, X_fft.shape)
            X_fft_noisy = X_fft + noise
            
            # Inverse FFT
            X_augmented = np.real(ifft(X_fft_noisy, axis=1))
            nadata.X = X_augmented
            
        elif method == "noise_injection":
            # Noise injection
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