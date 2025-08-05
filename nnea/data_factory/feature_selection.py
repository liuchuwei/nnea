"""
特征选择模块（na.fs）
提供多种特征选择方法，专门用于基因表达数据的特征选择
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict, Any
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression, RFE, RFECV,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


class fs:
    """
    特征选择类，提供多种特征选择方法
    """
    
    @staticmethod
    def select_features(nadata, method: str = "variance", n_features: int = 100, 
                       target_col: str = "target", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征选择主函数
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            特征选择方法：
            - 'variance': 方差选择
            - 'correlation': 相关性选择
            - 'mutual_info': 互信息选择
            - 'anova': 方差分析选择
            - 'lasso': Lasso回归选择
            - 'random_forest': 随机森林选择
            - 'rfe': 递归特征消除
            - 'rfecv': 交叉验证递归特征消除
            - 'select_from_model': 基于模型的特征选择
            - 'differential_expression': 差异表达分析
            - 'survival_analysis': 生存分析选择
        n_features : int
            选择的特征数量，默认100
        target_col : str
            目标变量列名
        **kwargs : 
            其他参数，包括：
            - need_scaling: bool, 是否需要在特征选择时进行标准化（默认True）
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (X_selected, feature_indices) 选择后的特征矩阵和特征索引
        """
        if nadata.X is None:
            raise ValueError("Expression matrix X is None")
        
        X = nadata.X
        y = nadata.Meta.get(target_col, None)
        
        if method == "variance":
            return fs._variance_selection(X, n_features, **kwargs)
        elif method == "correlation":
            return fs._correlation_selection(X, y, n_features, **kwargs)
        elif method == "mutual_info":
            return fs._mutual_info_selection(X, y, n_features, **kwargs)
        elif method == "anova":
            return fs._anova_selection(X, y, n_features, **kwargs)
        elif method == "lasso":
            return fs._lasso_selection(X, y, n_features, **kwargs)
        elif method == "random_forest":
            return fs._random_forest_selection(X, y, n_features, **kwargs)
        elif method == "rfe":
            return fs._rfe_selection(X, y, n_features, **kwargs)
        elif method == "rfecv":
            return fs._rfecv_selection(X, y, n_features, **kwargs)
        elif method == "select_from_model":
            return fs._select_from_model_selection(X, y, n_features, **kwargs)
        elif method == "differential_expression":
            return fs._differential_expression_selection(nadata, n_features, **kwargs)
        elif method == "survival_analysis":
            return fs._survival_analysis_selection(nadata, n_features, **kwargs)
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
    
    @staticmethod
    def _variance_selection(X: np.ndarray, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """方差选择"""
        # 计算方差
        variances = np.var(X, axis=1)
        top_indices = np.argsort(variances)[-n_features:]
        return X[top_indices, :], top_indices
    
    @staticmethod
    def _correlation_selection(X: np.ndarray, y: pd.Series, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """相关性选择"""
        if y is None:
            raise ValueError("Target variable y is required for correlation selection")
        
        # 计算每个特征与目标变量的相关性
        correlations = []
        for i in range(X.shape[0]):
            corr = np.corrcoef(X[i, :], y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        # 选择相关性最高的特征
        top_indices = np.argsort(correlations)[-n_features:]
        return X[top_indices, :], top_indices
    
    @staticmethod
    def _mutual_info_selection(X: np.ndarray, y: pd.Series, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """互信息选择"""
        if y is None:
            raise ValueError("Target variable y is required for mutual info selection")
        
        # 使用互信息进行特征选择
        if len(np.unique(y)) == 2:  # 分类问题
            selector = SelectKBest(mutual_info_classif, k=n_features)
        else:  # 回归问题
            selector = SelectKBest(mutual_info_regression, k=n_features)
        
        X_selected = selector.fit_transform(X.T, y).T
        feature_indices = np.where(selector.get_support())[0]
        return X_selected, feature_indices
    
    @staticmethod
    def _anova_selection(X: np.ndarray, y: pd.Series, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """方差分析选择"""
        if y is None:
            raise ValueError("Target variable y is required for ANOVA selection")
        
        # 使用F检验进行特征选择
        if len(np.unique(y)) == 2:  # 分类问题
            selector = SelectKBest(f_classif, k=n_features)
        else:  # 回归问题
            selector = SelectKBest(f_regression, k=n_features)
        
        X_selected = selector.fit_transform(X.T, y).T
        feature_indices = np.where(selector.get_support())[0]
        return X_selected, feature_indices
    
    @staticmethod
    def _lasso_selection(X: np.ndarray, y: pd.Series, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Lasso回归选择"""
        if y is None:
            raise ValueError("Target variable y is required for Lasso selection")

        # 使用Lasso进行特征选择
        alpha = kwargs.get('alpha', 0.01)
        if len(np.unique(y)) == 2:  # 分类问题
            model = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear', random_state=42)
        else:  # 回归问题
            model = Lasso(alpha=alpha, random_state=42)
        
        model.fit(X, y)
        
        # 获取非零系数的特征
        if hasattr(model, 'coef_'):
            coef = model.coef_
        else:
            coef = model.feature_importances_
        
        # 选择系数绝对值最大的特征
        feature_scores = np.abs(coef)
        top_indices = np.argsort(feature_scores)[0][-n_features:]
        return X[:, top_indices], top_indices
    
    @staticmethod
    def _random_forest_selection(X: np.ndarray, y: pd.Series, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """随机森林特征选择"""
        if y is None:
            raise ValueError("Target variable y is required for Random Forest selection")
        
        # 使用随机森林进行特征选择
        if len(np.unique(y)) == 2:  # 分类问题
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # 回归问题
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X, y)
        
        # 获取特征重要性
        feature_importance = model.feature_importances_
        top_indices = np.argsort(feature_importance)[-n_features:]
        return X[:, top_indices], top_indices
    
    @staticmethod
    def _rfe_selection(X: np.ndarray, y: pd.Series, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """递归特征消除"""
        if y is None:
            raise ValueError("Target variable y is required for RFE selection")
        
        # 使用线性SVM进行RFE
        estimator = LinearSVC(random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        
        X_selected = selector.fit_transform(X.T, y).T
        feature_indices = np.where(selector.support_)[0]
        return X_selected, feature_indices
    
    @staticmethod
    def _rfecv_selection(X: np.ndarray, y: pd.Series, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """交叉验证递归特征消除"""
        if y is None:
            raise ValueError("Target variable y is required for RFECV selection")
        
        # 使用线性SVM进行RFECV
        estimator = LinearSVC(random_state=42)
        selector = RFECV(estimator, min_features_to_select=n_features, step=1, cv=5)
        
        X_selected = selector.fit_transform(X.T, y).T
        feature_indices = np.where(selector.support_)[0]
        return X_selected, feature_indices
    
    @staticmethod
    def _select_from_model_selection(X: np.ndarray, y: pd.Series, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """基于模型的特征选择"""
        if y is None:
            raise ValueError("Target variable y is required for SelectFromModel selection")
        
        # 使用随机森林进行特征选择
        if len(np.unique(y)) == 2:  # 分类问题
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # 回归问题
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        selector = SelectFromModel(model, max_features=n_features)
        X_selected = selector.fit_transform(X.T, y).T
        feature_indices = np.where(selector.get_support())[0]
        return X_selected, feature_indices
    
    @staticmethod
    def _differential_expression_selection(nadata, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """差异表达分析选择"""
        X = nadata.X
        group_col = kwargs.get('group_col', 'target')
        method = kwargs.get('method', 't_test')
        
        if group_col not in nadata.Meta.columns:
            raise ValueError(f"Group column '{group_col}' not found in Meta data")
        
        groups = nadata.Meta[group_col]
        unique_groups = np.unique(groups)
        
        if len(unique_groups) != 2:
            raise ValueError("Differential expression analysis requires exactly 2 groups")
        
        # 计算差异表达统计量
        p_values = []
        for i in range(X.shape[1]):
            group1_data = X[groups == unique_groups[0], i]
            group2_data = X[groups == unique_groups[1], i]
            
            if method == "t_test":
                _, p_val = stats.ttest_ind(group1_data, group2_data)
            elif method == "mann_whitney":
                _, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            p_values.append(p_val)
        
        # 选择p值最小的特征
        top_indices = np.argsort(p_values)[:n_features]
        return X[:, top_indices], top_indices
    
    @staticmethod
    def _survival_analysis_selection(nadata, n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """生存分析特征选择"""
        X = nadata.X
        time_col = kwargs.get('time_col', 'time')
        event_col = kwargs.get('event_col', 'event')
        
        if time_col not in nadata.Meta.columns or event_col not in nadata.Meta.columns:
            raise ValueError(f"Time or event columns not found in Meta data")
        
        times = nadata.Meta[time_col]
        events = nadata.Meta[event_col]
        
        # 计算Cox比例风险模型的p值
        p_values = []
        for i in range(X.shape[0]):
            try:
                from lifelines import CoxPHFitter
                df = pd.DataFrame({
                    'time': times,
                    'event': events,
                    'expression': X[i, :]
                })
                cph = CoxPHFitter()
                cph.fit(df, duration_col='time', event_col='event')
                p_val = cph.print_summary().loc['expression', 'p']
                p_values.append(p_val)
            except:
                # 如果lifelines不可用，使用简单的相关性
                corr = np.corrcoef(X[i, :], times)[0, 1]
                p_values.append(1 - abs(corr))
        
        # 选择p值最小的特征
        top_indices = np.argsort(p_values)[:n_features]
        return X[top_indices, :], top_indices
    
    @staticmethod
    def apply_feature_selection(nadata, method: str = "variance", n_features: int = 100, 
                              target_col: str = "target", **kwargs) -> 'nadata':
        """
        对nadata对象应用特征选择
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            特征选择方法
        n_features : int
            选择的特征数量
        target_col : str
            目标变量列名
        **kwargs : 
            其他参数，包括：
            - need_scaling: bool, 是否需要在特征选择时进行标准化（默认True）
            
        Returns:
        --------
        nadata
            应用特征选择后的nadata对象
        """
        X_selected, feature_indices = fs.select_features(
            nadata, method, n_features, target_col, **kwargs
        )
        
        # 更新nadata对象
        nadata.X = X_selected
        
        # 更新基因信息
        if nadata.Var is not None:
            nadata.Var = nadata.Var.iloc[feature_indices]
        
        # 更新先验知识
        if nadata.Prior is not None:
            nadata.Prior = nadata.Prior[:, feature_indices]
        
        # 保存特征选择信息
        if not hasattr(nadata, 'feature_selection'):
            nadata.feature_selection = {}
        
        nadata.feature_selection.update({
            'method': method,
            'n_features': n_features,
            'selected_indices': feature_indices,
            'target_col': target_col,
            'kwargs': kwargs
        })
        
        return nadata
    
    @staticmethod
    def get_feature_importance(nadata, method: str = "random_forest", 
                             target_col: str = "target", **kwargs) -> Dict[str, Any]:
        """
        获取特征重要性
        
        Parameters:
        -----------
        nadata : nadata对象
            包含数据的nadata对象
        method : str
            特征重要性计算方法
        target_col : str
            目标变量列名
        **kwargs : 
            其他参数
            
        Returns:
        --------
        Dict[str, Any]
            特征重要性信息
        """
        X = nadata.X
        y = nadata.Meta.get(target_col, None)
        
        if y is None:
            raise ValueError("Target variable is required for feature importance calculation")
        
        if method == "random_forest":
            if len(np.unique(y)) == 2:  # 分类问题
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:  # 回归问题
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(X.T, y)
            importance = model.feature_importances_
            
        elif method == "correlation":
            importance = []
            for i in range(X.shape[0]):
                corr = np.corrcoef(X[i, :], y)[0, 1]
                importance.append(abs(corr) if not np.isnan(corr) else 0)
            importance = np.array(importance)
            
        elif method == "mutual_info":
            if len(np.unique(y)) == 2:  # 分类问题
                importance = mutual_info_classif(X.T, y)
            else:  # 回归问题
                importance = mutual_info_regression(X.T, y)
                
        else:
            raise ValueError(f"Unsupported importance method: {method}")
        
        # 创建特征重要性DataFrame
        if nadata.Var is not None:
            feature_names = nadata.Var.index.tolist()
        else:
            feature_names = [f"Gene_{i}" for i in range(X.shape[0])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return {
            'importance_df': importance_df,
            'method': method,
            'feature_names': feature_names,
            'importance_scores': importance
        } 