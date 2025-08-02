"""
评估指标模块
提供各种机器学习模型的评估指标
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Any, Union, List


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: np.ndarray = None, task: str = "classification") -> Dict[str, float]:
    """
    计算各种评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（用于分类任务）
        task: 任务类型 ("classification" 或 "regression")
    
    Returns:
        包含各种指标的字典
    """
    metrics = {}
    
    if task == "classification":
        # 分类指标
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm
        
    elif task == "regression":
        # 回归指标
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
    
    return metrics


def calculate_feature_importance(model, feature_names: List[str] = None) -> pd.DataFrame:
    """
    计算特征重要性
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
    
    Returns:
        特征重要性DataFrame
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if len(importances.shape) > 1:
            importances = np.mean(importances, axis=0)
    else:
        raise ValueError("模型不支持特征重要性计算")
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算相关性矩阵
    
    Args:
        data: 输入数据
    
    Returns:
        相关性矩阵
    """
    return data.corr()


def calculate_statistics(data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
    """
    计算基本统计信息
    
    Args:
        data: 输入数据
    
    Returns:
        统计信息字典
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    stats = {
        "shape": data.shape,
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "numeric_stats": data.describe().to_dict() if data.select_dtypes(include=[np.number]).shape[1] > 0 else {},
        "categorical_stats": {}
    }
    
    # 分类变量统计
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        stats["categorical_stats"][col] = data[col].value_counts().to_dict()
    
    return stats 