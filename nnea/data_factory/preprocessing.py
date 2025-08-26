"""
Data preprocessing module (na.pp)
Contains data standardization, missing value handling, outlier detection and processing, gene/sample filtering and other functions
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
    Data preprocessing class, providing various preprocessing methods
    """
    
    @staticmethod
    def process_survival_data(nadata, os_col: str = 'OS', os_time_col: str = 'OS.time', 
                              time_unit: str = 'auto'):
        """
        Standard survival data processing
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing survival data
        os_col : str
            Survival status column name, default is 'OS'
        os_time_col : str
            Survival time column name, default is 'OS.time'
        time_unit : str
            Time unit: 'auto', 'days', 'months', 'years'
            If 'auto', will automatically determine and convert to months
            
        Returns:
        --------
        nadata
            Processed nadata object
        """
        if nadata.Meta is None:
            raise ValueError("nadata.Meta is None, cannot process survival data")
        
        if os_col not in nadata.Meta.columns:
            raise ValueError(f"Column '{os_col}' not found in nadata.Meta")
        
        if os_time_col not in nadata.Meta.columns:
            raise ValueError(f"Column '{os_time_col}' not found in nadata.Meta")
        
        # Extract survival data
        y = nadata.Meta.loc[:, [os_col, os_time_col]].copy()
        
        # Handle survival time unit conversion
        os_time = y[os_time_col]
        
        if time_unit == 'auto':
            # Automatically determine time unit and convert to months
            max_time = os_time.max()
            if max_time > 1000:
                # Assume days, convert to months
                y[os_time_col] = os_time / 30.44
                print(f"🕐 Detected time unit as days, converted to months (divided by 30.44)")
            elif max_time < 100:
                # Assume years, convert to months
                y[os_time_col] = os_time * 12
                print(f"🕐 Detected time unit as years, converted to months (multiplied by 12)")
            else:
                # Already in months, no processing needed
                print(f"🕐 Detected time unit as months, no conversion needed")
        elif time_unit == 'days':
            y[os_time_col] = os_time / 30.44
            print(f"🕐 Converted time from days to months (divided by 30.44)")
        elif time_unit == 'years':
            y[os_time_col] = os_time * 12
            print(f"🕐 Converted time from years to months (multiplied by 12)")
        elif time_unit == 'months':
            # Already in months, no processing needed
            pass
        else:
            raise ValueError(f"Unsupported time_unit: {time_unit}")
        
        # Handle survival status labels
        os_col_data = y[os_col]
        
        # Check if OS is 0/1 variable, convert if string
        if os_col_data.dtype == object or str(os_col_data.dtype).startswith('str'):
            # Common survival analysis label mapping
            label_mapping = {
                'Dead': 1, 'Alive': 0,
                'deceased': 1, 'living': 0,
                '1': 1, '0': 0,
                'TRUE': 1, 'FALSE': 0,
                'True': 1, 'False': 0,
                'T': 1, 'F': 0,
                't': 1, 'f': 0
            }
            
            # Try mapping
            y[os_col] = os_col_data.map(label_mapping)
            
            # Check for unmapped values
            if y[os_col].isnull().any():
                try:
                    # Try direct conversion to int
                    y[os_col] = os_col_data.astype(int)
                    print(f"🏷️ Converted survival status from string to numeric")
                except Exception as e:
                    # Show unmapped values
                    unmapped_values = os_col_data[y[os_col].isnull()].unique()
                    raise ValueError(f"OS column cannot be converted to 0/1 variable, unmapped values: {unmapped_values}")
            else:
                print(f"🏷️ Converted survival status from string to numeric (mapping successful)")
        else:
            # If already numeric, ensure 0/1 format
            y[os_col] = os_col_data.apply(lambda v: 1 if v == 1 else 0)
            print(f"🏷️ Survival status already numeric, ensuring 0/1 format")
        
        # Validate processing results
        unique_os_values = y[os_col].unique()
        if not all(val in [0, 1] for val in unique_os_values):
            raise ValueError(f"Survival status processing failed, contains non-0/1 values: {unique_os_values}")
        
        # Check if time values are reasonable
        if (y[os_time_col] < 0).any():
            warnings.warn("Detected negative survival time values")
        
        # Add processed data to nadata.Meta
        # Only assign survival status column to target_col
        nadata.Meta['Event'] = y[os_col]
    
        # Update original survival time column
        nadata.Meta['Time'] = y[os_time_col]
        
        print(f"✅ Survival data processing completed")
        print(f"   - Survival status: {os_col} -> Event")
        print(f"   - Survival time: '{os_time_col}'-> Time (unit: months)")
        print(f"   - Data shape: {nadata.Meta['Event'].shape}")
        print(f"   - Survival status distribution: {nadata.Meta['Event'].value_counts().to_dict()}")
        return nadata
    
    @staticmethod
    def fillna(X, method: str = "mean", fill_value: float = 0):
        """
        Handle missing values
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        method : str
            Processing method: 'mean', 'median', 'zero', 'drop'
        fill_value : float
            Fill value, used for 'zero' method
            
        Returns:
        --------
        np.ndarray
            Processed data matrix
        """
        if X is None:
            return X
        
        if not np.isnan(X).any():
            return X
        
        if method == "mean":
            # Check if each column is all NaN, if so fill with 0, otherwise fill with mean
            nan_all_col = np.isnan(X).all(axis=0)
            if nan_all_col.any():
                X[:, nan_all_col] = 0
            # Fill remaining columns with NaN using mean
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            
        elif method == "median":
            # Check if each column is all NaN, if so fill with 0, otherwise fill with median
            nan_all_col = np.isnan(X).all(axis=0)
            if nan_all_col.any():
                X[:, nan_all_col] = 0
            # Fill remaining columns with NaN using median
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            
        elif method == "zero":
            # Fill with specified value
            X = np.nan_to_num(X, nan=fill_value)
            
        elif method == "drop":
            # Delete rows containing missing values
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            
        else:
            raise ValueError(f"Unsupported fillna method: {method}")
        
        return X
    
    @staticmethod
    def scale(X, method: str = "standard"):
        """
        Data standardization
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        method : str
            Standardization method: 'standard', 'minmax', 'robust'
            
        Returns:
        --------
        np.ndarray
            Standardized data matrix
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
        Get X data for training and test sets
        
        Parameters:
        -----------
        X : np.ndarray
            Feature data matrix
        nadata : nadata object
            nadata object containing split information
        test_size : float
            Test set proportion
        random_state : int
            Random seed
            
        Returns:
        --------
        tuple
            (X_train, X_test)
        """
        if hasattr(nadata, 'Model') and hasattr(nadata.Model, 'indices'):
            # Use saved split information
            train_idx = nadata.Model.indices['train']
            test_idx = nadata.Model.indices['test']
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            # Use random split
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
        Get y data for training and test sets
        
        Parameters:
        -----------
        y : pd.Series or np.ndarray
            Label data
        nadata : nadata object
            nadata object containing split information
        test_size : float
            Test set proportion
        random_state : int
            Random seed
            
        Returns:
        --------
        tuple
            (y_train, y_test)
        """
        if hasattr(nadata, 'Model') and hasattr(nadata.Model, 'indices'):
            # Use saved split information
            train_idx = nadata.Model.indices['train']
            test_idx = nadata.Model.indices['test']
            if isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]
        else:
            # Use random split
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
        Data standardization
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing expression matrix
        method : str
            Standardization method: 'zscore', 'minmax', 'robust', 'quantile', 'cell_by_gene'
        scale_factor : float
            Scale factor, used for cell_by_gene method
            
        Returns:
        --------
        nadata
            Standardized nadata object
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
            # Quantile standardization
            X_normalized = np.zeros_like(X)
            for i in range(X.shape[0]):
                gene_exp = X[i, :]
                q75, q25 = np.percentile(gene_exp, [75, 25])
                if q75 != q25:
                    X_normalized[i, :] = (gene_exp - q25) / (q75 - q25)
                else:
                    X_normalized[i, :] = gene_exp
                    
        elif method == "cell_by_gene":
            # Standardize by cell (commonly used for single-cell data)
            X_normalized = _normalize_cell_by_gene(X, scale_factor)
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        nadata.X = X_normalized
        return nadata
    
    @staticmethod
    def handle_missing_values(nadata, method: str = "drop", fill_value: float = 0):
        """
        Missing value handling
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Processing method: 'drop', 'fill', 'interpolate'
        fill_value : float
            Fill value, used for fill method
            
        Returns:
        --------
        nadata
            Processed nadata object
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X
        
        if method == "drop":
            # Delete genes or samples containing missing values
            # Delete genes (rows)
            gene_mask = ~np.isnan(X).any(axis=1)
            X_clean = X[gene_mask, :]
            if nadata.Var is not None:
                nadata.Var = nadata.Var.iloc[gene_mask]
            
            # Delete samples (columns)
            sample_mask = ~np.isnan(X_clean).any(axis=0)
            X_clean = X_clean[:, sample_mask]
            if nadata.Meta is not None:
                nadata.Meta = nadata.Meta.iloc[sample_mask]
            
            nadata.X = X_clean
            
        elif method == "fill":
            # Fill with specified value
            X_filled = np.nan_to_num(X, nan=fill_value)
            nadata.X = X_filled
            
        elif method == "interpolate":
            # Interpolation filling
            from scipy.interpolate import interp1d
            
            X_interpolated = X.copy()
            for i in range(X.shape[0]):
                gene_exp = X[i, :]
                if np.isnan(gene_exp).any():
                    # Find indices of non-missing values
                    valid_idx = ~np.isnan(gene_exp)
                    if valid_idx.sum() > 1:
                        # Interpolate
                        f = interp1d(np.where(valid_idx)[0], gene_exp[valid_idx], 
                                    kind='linear', fill_value='extrapolate')
                        all_idx = np.arange(len(gene_exp))
                        X_interpolated[i, :] = f(all_idx)
                    else:
                        # If only one valid value, fill with that value
                        valid_value = gene_exp[valid_idx][0]
                        X_interpolated[i, :] = valid_value
            
            nadata.X = X_interpolated
            
        else:
            raise ValueError(f"Unsupported missing value method: {method}")
        
        return nadata
    
    @staticmethod
    def detect_outliers(nadata, method: str = "iqr", threshold: float = 1.5):
        """
        Outlier detection
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Detection method: 'iqr', 'zscore', 'isolation_forest'
        threshold : float
            Threshold
            
        Returns:
        --------
        nadata
            Processed nadata object
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X
        outlier_mask = np.zeros(X.shape, dtype=bool)
        
        if method == "iqr":
            # IQR method
            for i in range(X.shape[0]):
                gene_exp = X[i, :]
                Q1 = np.percentile(gene_exp, 25)
                Q3 = np.percentile(gene_exp, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask[i, :] = (gene_exp < lower_bound) | (gene_exp > upper_bound)
                
        elif method == "zscore":
            # Z-score method
            for i in range(X.shape[0]):
                gene_exp = X[i, :]
                z_scores = np.abs((gene_exp - np.mean(gene_exp)) / np.std(gene_exp))
                outlier_mask[i, :] = z_scores > threshold
                
        elif method == "isolation_forest":
            # Isolation forest method
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X.T)
            outlier_mask = (outlier_labels == -1).reshape(X.shape[1], X.shape[0]).T
            
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        # Set outliers to NaN
        X_clean = X.copy()
        X_clean[outlier_mask] = np.nan
        nadata.X = X_clean
        
        return nadata
    
    @staticmethod
    def filter_genes(nadata, method: str = "variance", **kwargs):
        """
        Gene filtering
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Filtering method: 'variance', 'top_k', 'expression_threshold'
        **kwargs : 
            Other parameters
            
        Returns:
        --------
        nadata
            Filtered nadata object
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X
        
        if method == "variance":
            # Variance filtering
            threshold = kwargs.get('threshold', 0.01)
            selector = VarianceThreshold(threshold=threshold)
            X_filtered = selector.fit_transform(X.T).T
            gene_mask = selector.get_support()
            
        elif method == "top_k":
            # Select top k genes
            k = kwargs.get('k', 1000)
            if k >= X.shape[0]:
                return nadata
            
            # Calculate variance
            variances = np.var(X, axis=1)
            top_indices = np.argsort(variances)[-k:]
            X_filtered = X[top_indices, :]
            gene_mask = np.zeros(X.shape[0], dtype=bool)
            gene_mask[top_indices] = True
            
        elif method == "expression_threshold":
            # Expression threshold filtering
            threshold = kwargs.get('threshold', 0)
            min_cells = kwargs.get('min_cells', 1)
            
            # Calculate how many cells each gene is expressed in
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
        Sample filtering
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        method : str
            Filtering method: 'quality', 'expression_threshold'
        **kwargs : 
            Other parameters
            
        Returns:
        --------
        nadata
            Filtered nadata object
        """
        if nadata.X is None:
            return nadata
        
        X = nadata.X
        
        if method == "quality":
            # Quality filtering
            min_genes = kwargs.get('min_genes', 1)
            max_genes = kwargs.get('max_genes', float('inf'))
            
            # Calculate number of expressed genes per sample
            expressed_genes = (X > 0).sum(axis=0)
            sample_mask = (expressed_genes >= min_genes) & (expressed_genes <= max_genes)
            
        elif method == "expression_threshold":
            # Expression threshold filtering
            threshold = kwargs.get('threshold', 0)
            min_genes = kwargs.get('min_genes', 1)
            
            # Calculate number of genes with expression above threshold per sample
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
        Data splitting
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
        test_size : float
            Test set proportion
        val_size : float
            Validation set proportion
        random_state : int
            Random seed
        strategy : str
            Split strategy: 'random', 'stratified'
            
        Returns:
        --------
        nadata
            nadata object containing split information
        """
        if nadata.X is None:
            return nadata
        
        n_samples = nadata.X.shape[0]
        indices = np.arange(n_samples)
        
        if strategy == "random":
            from sklearn.model_selection import train_test_split
            
            # First split out test set
            train_indices, test_indices = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )

            
        elif strategy == "stratified":
            from sklearn.model_selection import train_test_split
            
            # Need target variable for stratified sampling
            if nadata.Meta is not None and 'target' in nadata.Meta.columns:
                target = nadata.Meta['target']
                
                # First split out test set
                train_indices, test_indices = train_test_split(
                    indices, test_size=test_size, random_state=random_state, 
                    stratify=target
                )

            else:
                warnings.warn("No target column found for stratified sampling, using random split")
                return pp.split_data(nadata, test_size, random_state, "random")
        
        # Save split information to nadata.Model.indices
        nadata.Model.set_indices(
            train_idx=train_indices,
            test_idx=test_indices,
        )
        
        # Also save strategy information to config
        if not hasattr(nadata, 'config'):
            nadata.config = {}
        nadata.config['data_split_strategy'] = strategy
        
        return nadata


def _normalize_cell_by_gene(X: np.ndarray, scale_factor: float = 10000) -> np.ndarray:
    """
    Standardize by cell (commonly used for single-cell data)
    
    Parameters:
    -----------
    X : np.ndarray
        Expression matrix
    scale_factor : float
        Scale factor
        
    Returns:
    --------
    np.ndarray
        Standardized expression matrix
    """
    # Calculate total expression per cell
    cell_sums = np.sum(X, axis=0)
    
    # Standardize
    X_normalized = X / cell_sums * scale_factor
    
    # Log transformation
    X_normalized = np.log1p(X_normalized)
    
    return X_normalized 