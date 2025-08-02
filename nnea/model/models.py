"""
NNEA模型工厂
根据配置选择不同的模型类型
"""

import logging
import torch
from typing import Dict, Any, Optional
from .base import BaseModel
from .nnea_model import NNEAClassifier

logger = logging.getLogger(__name__)

def build_model(config: Dict[str, Any]) -> BaseModel:
    """
    根据配置构建模型
    
    Args:
        config: 模型配置
        
    Returns:
        构建好的模型实例
    """
    model_type = config.get('global', {}).get('model', 'nnea')
    
    # 确保设备配置正确传递
    device_config = config.get('global', {}).get('device', 'cpu')
    if device_config == 'cuda' and torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        config['device'] = 'cpu'
    
    # 处理NNEA配置的展平
    if model_type == 'nnea' and 'nnea' in config:
        # 展平NNEA配置
        try:
            from utils.io_utils import flatten_dict
        except ImportError:
            # 如果导入失败，创建一个简单的flatten_dict函数
            def flatten_dict(d, parent_key='', sep='.'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
        
        nnea_config = flatten_dict(config['nnea'])
        # 合并配置
        model_config = {**config, **nnea_config}
    else:
        model_config = config
    
    if model_type == 'nnea':
        logger.info("构建NNEA分类器")
        return NNEAClassifier(model_config)
    elif model_type == 'nnea_regression':
        logger.info("构建NNEA回归器")
        # TODO: 实现NNEA回归器
        raise NotImplementedError("NNEA回归器尚未实现")
    elif model_type == 'nnea_survival':
        logger.info("构建NNEA生存分析模型")
        # TODO: 实现NNEA生存分析模型
        raise NotImplementedError("NNEA生存分析模型尚未实现")
    elif model_type == 'nnea_dimension':
        logger.info("构建NNEA降维模型")
        # TODO: 实现NNEA降维模型
        raise NotImplementedError("NNEA降维模型尚未实现")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def build(nadata) -> None:
    """
    构建模型并添加到nadata的Model容器中
    
    Args:
        nadata: nadata对象
    """
    if nadata is None:
        raise ValueError("nadata对象不能为空")
    
    # 获取模型配置
    config = nadata.Model.get_config()
    if not config:
        # 如果没有配置，尝试从nadata.config获取（向后兼容）
        config = getattr(nadata, 'config', {})
        if config:
            nadata.Model.set_config(config)
    
    model_type = config.get('global', {}).get('model', 'nnea')
    
    # 构建模型
    model = build_model(config)
    
    # 构建模型
    model.build(nadata)
    
    # 保存到nadata的Model容器
    nadata.Model.add_model(model_type, model)
    
    logger.info(f"模型已构建并添加到nadata.Model: {model_type}")

def train(nadata, model_name: Optional[str] = None, verbose: int = 1) -> Dict[str, Any]:
    """
    训练模型
    
    Args:
        nadata: nadata对象
        model_name: 模型名称，如果为None则使用默认模型
        verbose: 详细程度，0=只显示进度条，1=显示基本信息，2=显示详细评估结果
        
    Returns:
        训练结果
    """
    if not nadata.Model.models:
        raise ValueError("nadata.Model中没有模型，请先调用build()")
    
    # 确定要训练的模型
    if model_name is None:
        model_type = nadata.Model.get_config().get('global', {}).get('model', 'nnea')
        model = nadata.Model.get_model(model_type)
    else:
        model = nadata.Model.get_model(model_name)
    
    if model is None:
        raise ValueError(f"未找到模型: {model_name or 'default'}")
    
    # 训练模型
    train_results = model.train(nadata, verbose=verbose)
    
    # 保存训练结果到Model容器
    nadata.Model.set_train_results(train_results)
    
    return train_results

def eval(nadata, split='test', model_name: Optional[str] = None) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        nadata: nadata对象
        split: 评估的数据集分割
        model_name: 模型名称
        
    Returns:
        评估结果
    """
    if not nadata.Model.models:
        raise ValueError("nadata.Model中没有模型，请先调用build()")
    
    # 确定要评估的模型
    if model_name is None:
        model_type = nadata.Model.get_config().get('global', {}).get('model', 'nnea')
        model = nadata.Model.get_model(model_type)
    else:
        model = nadata.Model.get_model(model_name)
    
    if model is None:
        raise ValueError(f"未找到模型: {model_name or 'default'}")
    
    # 评估模型
    return model.evaluate(nadata, split)

def explain(nadata, method='importance', model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    解释模型
    
    Args:
        nadata: nadata对象
        method: 解释方法
        model_name: 模型名称
        
    Returns:
        解释结果
    """
    if not nadata.Model.models:
        raise ValueError("nadata.Model中没有模型，请先调用build()")
    
    # 确定要解释的模型
    if model_name is None:
        model_type = nadata.Model.get_config().get('global', {}).get('model', 'nnea')
        model = nadata.Model.get_model(model_type)
    else:
        model = nadata.Model.get_model(model_name)
    
    if model is None:
        raise ValueError(f"未找到模型: {model_name or 'default'}")
    
    # 解释模型
    return model.explain(nadata, method)

def save_model(nadata, save_path: str, model_name: Optional[str] = None) -> None:
    """
    保存模型或整个nadata项目
    
    Args:
        nadata: nadata对象
        save_path: 保存路径
        model_name: 模型名称，如果为None则保存整个项目
    """
    if model_name is None:
        # 保存整个项目
        nadata.save(save_path)
        logger.info(f"项目已保存到: {save_path}")
    else:
        # 保存特定模型
        model = nadata.Model.get_model(model_name)
        if model is None:
            raise ValueError(f"未找到模型: {model_name}")
        
        # 保存模型状态
        torch.save(model.state_dict(), save_path)
        logger.info(f"模型 {model_name} 已保存到: {save_path}")

def load_project(load_path: str):
    """
    加载nadata项目
    
    Args:
        load_path: 加载路径
        
    Returns:
        nadata对象
    """
    from ..io._load import load_project as load_project_impl
    return load_project_impl(load_path)

def get_summary(nadata) -> Dict[str, Any]:
    """
    获取nadata摘要信息
    
    Args:
        nadata: nadata对象
        
    Returns:
        摘要信息字典
    """
    summary = {
        'data_info': {},
        'model_info': {},
        'config_info': {}
    }
    
    # 数据信息
    if nadata.X is not None:
        summary['data_info']['X_shape'] = nadata.X.shape
    if nadata.Meta is not None:
        summary['data_info']['Meta_shape'] = nadata.Meta.shape
    if nadata.Var is not None:
        summary['data_info']['Var_shape'] = nadata.Var.shape
    if nadata.Prior is not None:
        summary['data_info']['Prior_shape'] = nadata.Prior.shape
    
    # 模型信息
    summary['model_info']['models'] = nadata.Model.list_models()
    summary['model_info']['config_keys'] = list(nadata.Model.get_config().keys())
    summary['model_info']['train_results_keys'] = list(nadata.Model.get_train_results().keys())
    
    # 配置信息
    config = nadata.Model.get_config()
    if config:
        summary['config_info']['model_type'] = config.get('global', {}).get('model', 'unknown')
        summary['config_info']['task'] = config.get('global', {}).get('task', 'unknown')
        summary['config_info']['device'] = config.get('global', {}).get('device', 'cpu')
    
    return summary

def train_classification_models(nadata, config: Optional[Dict[str, Any]] = None, verbose: int = 1) -> Dict[str, Any]:
    """
    训练分类模型
    
    Args:
        nadata: nadata对象
        config: 配置字典
        verbose: 详细程度
        
    Returns:
        训练结果
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import numpy as np
    
    # 获取配置
    if config is None:
        config = nadata.Model.get_config()
    
    # 获取数据
    X_train = nadata.X[:, nadata.Model.get_indices('train')]
    X_test = nadata.X[:, nadata.Model.get_indices('test')]
    
    # 获取目标列名称
    target_column = 'class'  # 默认使用'class'
    if config and 'dataset' in config:
        target_column = config['dataset'].get('target_column', 'class')
    
    y_train = nadata.Meta.iloc[nadata.Model.get_indices('train')][target_column].values
    y_test = nadata.Meta.iloc[nadata.Model.get_indices('test')][target_column].values
    
    # 定义要训练的模型
    models_to_train = config.get('classification', {}).get('models', ['logistic_regression', 'random_forest'])
    
    results = {
        'models': {},
        'comparison_df': None
    }
    
    for model_name in models_to_train:
        if verbose >= 1:
            print(f"训练 {model_name}...")
        
        if model_name == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'support_vector_machine':
            model = SVC(probability=True, random_state=42)
        else:
            if verbose >= 1:
                print(f"跳过未知模型: {model_name}")
            continue
        
        # 训练模型
        model.fit(X_train.T, y_train)
        
        # 预测
        y_pred = model.predict(X_test.T)
        y_pred_proba = model.predict_proba(X_test.T)[:, 1]
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 保存结果
        results['models'][model_name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # 添加到nadata的Model容器
        nadata.Model.add_model(f'classification_{model_name}', model)
        
        if verbose >= 1:
            print(f"  {model_name} - 准确率: {accuracy:.4f}, AUC: {auc:.4f}")
    
    # 创建比较DataFrame
    if results['models']:
        comparison_data = []
        for name, result in results['models'].items():
            comparison_data.append({
                'model': name,
                'accuracy': result['accuracy'],
                'auc': result['auc']
            })
        
        import pandas as pd
        results['comparison_df'] = pd.DataFrame(comparison_data)
    
    return results

def compare_models(nadata, config: Optional[Dict[str, Any]] = None, verbose: int = 1) -> Dict[str, Any]:
    """
    比较不同模型的性能
    
    Args:
        nadata: nadata对象
        config: 配置字典
        verbose: 详细程度
        
    Returns:
        比较结果
    """
    # 获取所有模型
    all_models = nadata.Model.list_models()
    
    if verbose >= 1:
        print(f"比较 {len(all_models)} 个模型: {all_models}")
    
    results = {
        'models': {},
        'comparison_df': None
    }
    
    # 获取测试数据
    test_indices = nadata.Model.get_indices('test')
    if test_indices is None:
        raise ValueError("没有找到测试集索引")
    
    X_test = nadata.X[:, test_indices]
    
    # 获取目标列名
    target_column = nadata.Model.get_config().get('dataset', {}).get('target_column', 'class')
    y_test = nadata.Meta.iloc[test_indices][target_column].values
    
    for model_name in all_models:
        model = nadata.Model.get_model(model_name)
        if model is None:
            continue
        
        if verbose >= 1:
            print(f"评估 {model_name}...")
        
        try:
            # 对于NNEA模型
            if hasattr(model, 'evaluate'):
                eval_result = model.evaluate(nadata, 'test')
                results['models'][model_name] = eval_result
            else:
                # 对于sklearn模型
                y_pred = model.predict(X_test.T)
                y_pred_proba = model.predict_proba(X_test.T)[:, 1]
                
                from sklearn.metrics import accuracy_score, roc_auc_score
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results['models'][model_name] = {
                    'accuracy': accuracy,
                    'auc': auc
                }
            
            if verbose >= 1:
                if 'accuracy' in results['models'][model_name]:
                    print(f"  {model_name} - 准确率: {results['models'][model_name]['accuracy']:.4f}")
                if 'auc' in results['models'][model_name]:
                    print(f"  {model_name} - AUC: {results['models'][model_name]['auc']:.4f}")
                    
        except Exception as e:
            if verbose >= 1:
                print(f"  评估 {model_name} 失败: {e}")
            continue
    
    # 创建比较DataFrame
    if results['models']:
        comparison_data = []
        for name, result in results['models'].items():
            row = {'model': name}
            if 'accuracy' in result:
                row['accuracy'] = result['accuracy']
            if 'auc' in result:
                row['auc'] = result['auc']
            comparison_data.append(row)
        
        import pandas as pd
        results['comparison_df'] = pd.DataFrame(comparison_data)
    
    return results 