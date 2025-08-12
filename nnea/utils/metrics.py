"""
Evaluation Metrics Module
Provides various evaluation metrics for machine learning models
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
    Calculate various evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for classification tasks)
        task: Task type ("classification" or "regression")
    
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    if task == "classification":
        # Classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm
        
    elif task == "regression":
        # Regression metrics
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
    
    return metrics


def calculate_feature_importance(model, feature_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate feature importance
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        Feature importance DataFrame
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if len(importances.shape) > 1:
            importances = np.mean(importances, axis=0)
    else:
        raise ValueError("Model does not support feature importance calculation")
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix
    
    Args:
        data: Input data
    
    Returns:
        Correlation matrix
    """
    return data.corr()


def calculate_statistics(data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
    """
    Calculate basic statistical information
    
    Args:
        data: Input data
    
    Returns:
        Statistical information dictionary
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
    
    # Categorical variable statistics
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        stats["categorical_stats"][col] = data[col].value_counts().to_dict()
    
    return stats 