"""
数据验证模块
包含数据完整性检查、数据一致性验证、格式验证等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings


class validation:
    """
    数据验证类，提供各种验证方法
    """
    
    @staticmethod
    def check_data_integrity(nadata) -> Dict[str, Any]:
        """
        检查数据完整性
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
            
        Returns:
        --------
        Dict[str, Any]
            完整性检查结果
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # 检查表达矩阵
        if nadata.X is None:
            results['errors'].append("Expression matrix X is None")
            results['is_valid'] = False
        else:
            # 检查NaN值
            nan_count = np.isnan(nadata.X).sum()
            if nan_count > 0:
                results['warnings'].append(f"Found {nan_count} NaN values in expression matrix")
            
            # 检查无穷值
            inf_count = np.isinf(nadata.X).sum()
            if inf_count > 0:
                results['errors'].append(f"Found {inf_count} infinite values in expression matrix")
                results['is_valid'] = False
            
            # 检查负值
            neg_count = (nadata.X < 0).sum()
            if neg_count > 0:
                results['warnings'].append(f"Found {neg_count} negative values in expression matrix")
            
            results['summary']['expression_matrix'] = {
                'shape': nadata.X.shape,
                'nan_count': nan_count,
                'inf_count': inf_count,
                'neg_count': neg_count
            }
        
        # 检查表型数据
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
        
        # 检查基因数据
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
        
        # 检查先验知识
        if nadata.Prior is not None:
            prior_shape = nadata.Prior.shape
            prior_nan_count = np.isnan(nadata.Prior).sum()
            
            if prior_nan_count > 0:
                results['errors'].append(f"Found {prior_nan_count} NaN values in prior knowledge")
                results['is_valid'] = False
            
            results['summary']['prior_knowledge'] = {
                'shape': prior_shape,
                'nan_count': prior_nan_count
            }
        
        return results
    
    @staticmethod
    def check_data_consistency(nadata) -> Dict[str, Any]:
        """
        检查数据一致性
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
            
        Returns:
        --------
        Dict[str, Any]
            一致性检查结果
        """
        results = {
            'is_consistent': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # 检查维度一致性
        if nadata.X is not None:
            n_genes, n_samples = nadata.X.shape
            
            # 检查基因数据维度
            if nadata.Var is not None:
                if len(nadata.Var) != n_genes:
                    results['errors'].append(f"Gene data dimension mismatch: {len(nadata.Var)} vs {n_genes}")
                    results['is_consistent'] = False
                else:
                    results['summary']['gene_consistency'] = "OK"
            
            # 检查表型数据维度
            if nadata.Meta is not None:
                if len(nadata.Meta) != n_samples:
                    results['errors'].append(f"Phenotype data dimension mismatch: {len(nadata.Meta)} vs {n_samples}")
                    results['is_consistent'] = False
                else:
                    results['summary']['phenotype_consistency'] = "OK"
            
            # 检查先验知识维度
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
        
        # 检查基因名称一致性
        if nadata.Var is not None and nadata.X is not None:
            if 'gene_name' in nadata.Var.columns:
                gene_names = nadata.Var['gene_name'].values
                if len(gene_names) != nadata.X.shape[0]:
                    results['warnings'].append("Gene name count mismatch with expression matrix")
        
        return results
    
    @staticmethod
    def validate_format(nadata) -> Dict[str, Any]:
        """
        验证数据格式
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
            
        Returns:
        --------
        Dict[str, Any]
            格式验证结果
        """
        results = {
            'is_valid_format': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # 验证表达矩阵格式
        if nadata.X is not None:
            if not isinstance(nadata.X, (np.ndarray, pd.DataFrame)):
                results['errors'].append("Expression matrix must be numpy array or pandas DataFrame")
                results['is_valid_format'] = False
            
            if nadata.X.ndim != 2:
                results['errors'].append("Expression matrix must be 2-dimensional")
                results['is_valid_format'] = False
            
            results['summary']['expression_format'] = "OK"
        
        # 验证表型数据格式
        if nadata.Meta is not None:
            if not isinstance(nadata.Meta, pd.DataFrame):
                results['errors'].append("Phenotype data must be pandas DataFrame")
                results['is_valid_format'] = False
            else:
                results['summary']['phenotype_format'] = "OK"
        
        # 验证基因数据格式
        if nadata.Var is not None:
            if not isinstance(nadata.Var, pd.DataFrame):
                results['errors'].append("Gene data must be pandas DataFrame")
                results['is_valid_format'] = False
            else:
                results['summary']['gene_format'] = "OK"
        
        # 验证先验知识格式
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
        检查数据质量
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
            
        Returns:
        --------
        Dict[str, Any]
            质量检查结果
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
        
        # 检查数据稀疏性
        zero_count = (X == 0).sum()
        total_elements = X.size
        sparsity = zero_count / total_elements
        
        if sparsity > 0.9:
            results['issues'].append(f"Data is very sparse ({sparsity:.2%} zeros)")
            results['quality_score'] *= 0.8
        elif sparsity > 0.7:
            results['warnings'].append(f"Data is moderately sparse ({sparsity:.2%} zeros)")
            results['quality_score'] *= 0.9
        
        # 检查数据范围
        data_range = np.ptp(X)
        if data_range < 1e-6:
            results['issues'].append("Data has very small range")
            results['quality_score'] *= 0.7
        
        # 检查异常值
        q1 = np.percentile(X, 25)
        q3 = np.percentile(X, 75)
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        outlier_count = (X > outlier_threshold).sum()
        outlier_ratio = outlier_count / total_elements
        
        if outlier_ratio > 0.1:
            results['issues'].append(f"High proportion of outliers ({outlier_ratio:.2%})")
            results['quality_score'] *= 0.8
        
        # 检查基因表达分布
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
        
        # 生成建议
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
        综合数据验证
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
            
        Returns:
        --------
        Dict[str, Any]
            综合验证结果
        """
        results = {
            'overall_valid': True,
            'integrity_check': validation.check_data_integrity(nadata),
            'consistency_check': validation.check_data_consistency(nadata),
            'format_check': validation.validate_format(nadata),
            'quality_check': validation.check_data_quality(nadata),
            'summary': {}
        }
        
        # 汇总结果
        if not results['integrity_check']['is_valid']:
            results['overall_valid'] = False
        
        if not results['consistency_check']['is_consistent']:
            results['overall_valid'] = False
        
        if not results['format_check']['is_valid_format']:
            results['overall_valid'] = False
        
        # 计算总体质量分数
        quality_score = results['quality_check']['quality_score']
        results['summary']['overall_quality_score'] = quality_score
        
        # 生成总体建议
        all_recommendations = []
        all_recommendations.extend(results['integrity_check'].get('warnings', []))
        all_recommendations.extend(results['consistency_check'].get('warnings', []))
        all_recommendations.extend(results['format_check'].get('warnings', []))
        all_recommendations.extend(results['quality_check'].get('recommendations', []))
        
        results['summary']['recommendations'] = all_recommendations
        
        return results 