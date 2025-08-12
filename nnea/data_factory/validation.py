"""
Data Validation Module
Contains data integrity checks, data consistency validation, format validation and other functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings


class validation:
    """
    Data validation class, provides various validation methods
    """
    
    @staticmethod
    def check_data_integrity(nadata) -> Dict[str, Any]:
        """
        Check data integrity
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
            
        Returns:
        --------
        Dict[str, Any]
            Integrity check results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check expression matrix
        if nadata.X is None:
            results['errors'].append("Expression matrix X is None")
            results['is_valid'] = False
        else:
            # Check NaN values
            nan_count = np.isnan(nadata.X).sum()
            if nan_count > 0:
                results['warnings'].append(f"Found {nan_count} NaN values in expression matrix")
            
            # Check infinite values
            inf_count = np.isinf(nadata.X).sum()
            if inf_count > 0:
                results['errors'].append(f"Found {inf_count} infinite values in expression matrix")
                results['is_valid'] = False
            
            # Check negative values
            neg_count = (nadata.X < 0).sum()
            if neg_count > 0:
                results['warnings'].append(f"Found {neg_count} negative values in expression matrix")
            
            results['summary']['expression_matrix'] = {
                'shape': nadata.X.shape,
                'nan_count': nan_count,
                'inf_count': inf_count,
                'neg_count': neg_count
            }
        
        # Check phenotype data
        if nadata.Meta is not None:
            meta_shape = nadata.Meta.shape
            meta_nan_count = nadata.Meta.isnull().sum().sum()
            
            if meta_nan_count > 0:
                results['warnings'].append(f"Found {meta_nan_count} missing values in phenotype data")
            
            results['summary']['phenotype_data'] = {
                'shape': meta_shape,
                'nan_count': meta_nan_count,
                'columns': list(nadata.Meta.columns)
            }
        
        # Check gene data
        if nadata.Var is not None:
            var_shape = nadata.Var.shape
            var_nan_count = nadata.Var.isnull().sum().sum()
            
            if var_nan_count > 0:
                results['warnings'].append(f"Found {var_nan_count} missing values in gene data")
            
            results['summary']['gene_data'] = {
                'shape': var_shape,
                'nan_count': var_nan_count,
                'columns': list(nadata.Var.columns)
            }
        
        # Check prior knowledge
        if nadata.Prior is not None:
            prior_shape = nadata.Prior.shape
            prior_nan_count = np.isnan(nadata.Prior).sum()
            
            if prior_nan_count > 0:
                results['errors'].append(f"Found {prior_nan_count} NaN values in prior knowledge")
                results['is_valid'] = False
            
            results['summary']['piror_knowledge'] = {
                'shape': prior_shape,
                'nan_count': prior_nan_count
            }
        
        return results
    
    @staticmethod
    def check_data_consistency(nadata) -> Dict[str, Any]:
        """
        Check data consistency
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
            
        Returns:
        --------
        Dict[str, Any]
            Consistency check results
        """
        results = {
            'is_consistent': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check dimension consistency
        if nadata.X is not None:
            n_genes, n_samples = nadata.X.shape
            
            # Check gene data dimensions
            if nadata.Var is not None:
                if len(nadata.Var) != n_genes:
                    results['errors'].append(f"Gene data dimension mismatch: {len(nadata.Var)} vs {n_genes}")
                    results['is_consistent'] = False
                else:
                    results['summary']['gene_consistency'] = "OK"
            
            # Check phenotype data dimensions
            if nadata.Meta is not None:
                if len(nadata.Meta) != n_samples:
                    results['errors'].append(f"Phenotype data dimension mismatch: {len(nadata.Meta)} vs {n_samples}")
                    results['is_consistent'] = False
                else:
                    results['summary']['phenotype_consistency'] = "OK"
            
            # Check prior knowledge dimensions
            if nadata.Prior is not None:
                prior_genes, prior_pathways = nadata.Prior.shape
                if prior_genes != n_genes:
                    results['errors'].append(f"Prior knowledge gene dimension mismatch: {prior_genes} vs {n_genes}")
                    results['is_consistent'] = False
                else:
                    results['summary']['prior_consistency'] = "OK"
            
            results['summary']['dimensions'] = {
                'genes': n_genes,
                'samples': n_samples
            }
        
        # Check gene name consistency
        if nadata.Var is not None and nadata.X is not None:
            if 'gene_name' in nadata.Var.columns:
                gene_names = nadata.Var['gene_name'].values
                if len(gene_names) != nadata.X.shape[0]:
                    results['warnings'].append("Gene name count mismatch with expression matrix")
        
        return results
    
    @staticmethod
    def validate_format(nadata) -> Dict[str, Any]:
        """
        Validate data format
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
            
        Returns:
        --------
        Dict[str, Any]
            Format validation results
        """
        results = {
            'is_valid_format': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Validate expression matrix format
        if nadata.X is not None:
            if not isinstance(nadata.X, (np.ndarray, pd.DataFrame)):
                results['errors'].append("Expression matrix must be numpy array or pandas DataFrame")
                results['is_valid_format'] = False
            
            if nadata.X.ndim != 2:
                results['errors'].append("Expression matrix must be 2-dimensional")
                results['is_valid_format'] = False
            
            results['summary']['expression_format'] = "OK"
        
        # Validate phenotype data format
        if nadata.Meta is not None:
            if not isinstance(nadata.Meta, pd.DataFrame):
                results['errors'].append("Phenotype data must be pandas DataFrame")
                results['is_valid_format'] = False
            else:
                results['summary']['phenotype_format'] = "OK"
        
        # Validate gene data format
        if nadata.Var is not None:
            if not isinstance(nadata.Var, pd.DataFrame):
                results['errors'].append("Gene data must be pandas DataFrame")
                results['is_valid_format'] = False
            else:
                results['summary']['gene_format'] = "OK"
        
        # Validate prior knowledge format
        if nadata.Prior is not None:
            if not isinstance(nadata.Prior, (np.ndarray, pd.DataFrame)):
                results['errors'].append("Prior knowledge must be numpy array or pandas DataFrame")
                results['is_valid_format'] = False
            else:
                results['summary']['prior_format'] = "OK"
        
        return results
    
    @staticmethod
    def check_data_quality(nadata) -> Dict[str, Any]:
        """
        Check data quality
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
            
        Returns:
        --------
        Dict[str, Any]
            Quality check results
        """
        results = {
            'quality_score': 1.0,
            'issues': [],
            'recommendations': [],
            'summary': {}
        }
        
        if nadata.X is None:
            results['quality_score'] = 0.0
            results['issues'].append("No expression matrix provided")
            return results
        
        X = nadata.X
        
        # Check data sparsity
        zero_count = (X == 0).sum()
        total_elements = X.size
        sparsity = zero_count / total_elements
        
        if sparsity > 0.9:
            results['issues'].append(f"Data is very sparse ({sparsity:.2%} zeros)")
            results['quality_score'] *= 0.8
        elif sparsity > 0.7:
            results['warnings'].append(f"Data is moderately sparse ({sparsity:.2%} zeros)")
            results['quality_score'] *= 0.9
        
        # Check data range
        data_range = np.ptp(X)
        if data_range < 1e-6:
            results['issues'].append("Data has very small range")
            results['quality_score'] *= 0.7
        
        # Check outliers
        q1 = np.percentile(X, 25)
        q3 = np.percentile(X, 75)
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        outlier_count = (X > outlier_threshold).sum()
        outlier_ratio = outlier_count / total_elements
        
        if outlier_ratio > 0.1:
            results['issues'].append(f"High proportion of outliers ({outlier_ratio:.2%})")
            results['quality_score'] *= 0.8
        
        # Check gene expression distribution
        gene_means = np.mean(X, axis=1)
        gene_vars = np.var(X, axis=1)
        
        low_variance_genes = (gene_vars < np.percentile(gene_vars, 10)).sum()
        if low_variance_genes > X.shape[0] * 0.5:
            results['warnings'].append(f"Many genes have low variance ({low_variance_genes} genes)")
            results['quality_score'] *= 0.9
        
        results['summary'] = {
            'sparsity': sparsity,
            'data_range': data_range,
            'outlier_ratio': outlier_ratio,
            'low_variance_genes': low_variance_genes
        }
        
        # Generate recommendations
        if sparsity > 0.7:
            results['recommendations'].append("Consider using sparse matrix format")
        
        if outlier_ratio > 0.05:
            results['recommendations'].append("Consider outlier detection and removal")
        
        if low_variance_genes > X.shape[0] * 0.3:
            results['recommendations'].append("Consider filtering low-variance genes")
        
        return results
    
    @staticmethod
    def comprehensive_validation(nadata) -> Dict[str, Any]:
        """
        Comprehensive data validation
        
        Parameters:
        -----------
        nadata : nadata object
            nadata object containing data
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive validation results
        """
        results = {
            'overall_valid': True,
            'integrity_check': validation.check_data_integrity(nadata),
            'consistency_check': validation.check_data_consistency(nadata),
            'format_check': validation.validate_format(nadata),
            'quality_check': validation.check_data_quality(nadata),
            'summary': {}
        }
        
        # Summarize results
        if not results['integrity_check']['is_valid']:
            results['overall_valid'] = False
        
        if not results['consistency_check']['is_consistent']:
            results['overall_valid'] = False
        
        if not results['format_check']['is_valid_format']:
            results['overall_valid'] = False
        
        # Calculate overall quality score
        quality_score = results['quality_check']['quality_score']
        results['summary']['overall_quality_score'] = quality_score
        
        # Generate overall recommendations
        all_recommendations = []
        all_recommendations.extend(results['integrity_check'].get('warnings', []))
        all_recommendations.extend(results['consistency_check'].get('warnings', []))
        all_recommendations.extend(results['format_check'].get('warnings', []))
        all_recommendations.extend(results['quality_check'].get('recommendations', []))
        
        results['summary']['recommendations'] = all_recommendations
        
        return results 