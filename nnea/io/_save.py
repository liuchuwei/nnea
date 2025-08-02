import os
import torch
import logging
from typing import Dict, Any, Optional
# 避免循环导入，使用类型注解
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._nadata import nadata

# 获取logger
logger = logging.getLogger(__name__)


def save_project(nadata_obj, filepath: str, save_data: bool = True) -> None:
    """
    保存nadata项目到文件
    
    Parameters:
    -----------
    nadata_obj : nadata
        要保存的nadata对象
    filepath : str
        保存路径
    save_data : bool
        是否保存数据，如果为False只保存模型和配置
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 准备保存的数据
    checkpoint = {}
    
    # 保存配置（从Model容器获取）
    config = nadata_obj.Model.get_config()
    if config:
        checkpoint['config'] = config
    
    # 保存核心数据（可选）
    if save_data:
        if nadata_obj.X is not None:
            checkpoint['X'] = nadata_obj.X
        if nadata_obj.Meta is not None:
            checkpoint['Meta'] = nadata_obj.Meta
        if nadata_obj.Var is not None:
            checkpoint['Var'] = nadata_obj.Var
        if nadata_obj.Prior is not None:
            checkpoint['Prior'] = nadata_obj.Prior
    
    # 保存模型信息（从Model容器获取）
    if nadata_obj.Model:
        # 保存所有模型的状态字典
        model_states = {}
        for model_name, model in nadata_obj.Model.models.items():
            if hasattr(model, 'state_dict'):
                model_states[model_name] = model.state_dict()
            elif hasattr(model, 'get_params'):
                # 对于sklearn模型，保存参数
                model_states[model_name] = model.get_params()
        
        if model_states:
            checkpoint['model_states'] = model_states
        
        # 训练历史
        train_results = nadata_obj.Model.get_train_results()
        if train_results:
            checkpoint['train_results'] = train_results
        
        # 数据索引
        indices = nadata_obj.Model.get_indices()
        if any(indices.values()):
            checkpoint['indices'] = indices
        
        # 元数据
        metadata = nadata_obj.Model.get_metadata()
        if metadata:
            checkpoint['metadata'] = metadata
    
    # 保存到文件 - 确保兼容性
    torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
    
    logger.info(f"项目已保存: {filepath}")
    logger.info(f"保存的数据: {list(checkpoint.keys())}")


def save_model_only(nadata_obj, filepath: str) -> None:
    """
    只保存模型，不保存数据
    
    Parameters:
    -----------
    nadata_obj : nadata
        要保存的nadata对象
    filepath : str
        保存路径
    """
    save_project(nadata_obj, filepath, save_data=False)


def save_data_only(nadata_obj, filepath: str) -> None:
    """
    只保存数据，不保存模型
    
    Parameters:
    -----------
    nadata_obj : nadata
        要保存的nadata对象
    filepath : str
        保存路径
    """
    # 准备保存的数据
    checkpoint = {}
    
    # 保存核心数据
    if nadata_obj.X is not None:
        checkpoint['X'] = nadata_obj.X
    if nadata_obj.Meta is not None:
        checkpoint['Meta'] = nadata_obj.Meta
    if nadata_obj.Var is not None:
        checkpoint['Var'] = nadata_obj.Var
    if nadata_obj.Prior is not None:
        checkpoint['Prior'] = nadata_obj.Prior
    
    # 保存数据索引
    indices = nadata_obj.Model.get_indices()
    if any(indices.values()):
        checkpoint['indices'] = indices
    
    # 保存到文件
    torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
    
    logger.info(f"数据已保存: {filepath}")


def save_checkpoint(nadata_obj, filepath: str, epoch: int, 
                   save_data: bool = False, **kwargs) -> None:
    """
    保存检查点
    
    Parameters:
    -----------
    nadata_obj : nadata
        要保存的nadata对象
    filepath : str
        保存路径
    epoch : int
        当前epoch
    save_data : bool
        是否保存数据
    **kwargs : 其他参数
    """
    # 准备检查点数据
    checkpoint = {
        'epoch': epoch,
        'config': nadata_obj.Model.get_config(),
        'train_results': nadata_obj.Model.get_train_results(),
        'indices': nadata_obj.Model.get_indices(),
        'metadata': nadata_obj.Model.get_metadata()
    }
    
    # 保存模型状态
    model_states = {}
    for model_name, model in nadata_obj.Model.models.items():
        if hasattr(model, 'state_dict'):
            model_states[model_name] = model.state_dict()
    
    if model_states:
        checkpoint['model_states'] = model_states
    
    # 保存核心数据（可选）
    if save_data:
        if nadata_obj.X is not None:
            checkpoint['X'] = nadata_obj.X
        if nadata_obj.Meta is not None:
            checkpoint['Meta'] = nadata_obj.Meta
        if nadata_obj.Var is not None:
            checkpoint['Var'] = nadata_obj.Var
        if nadata_obj.Prior is not None:
            checkpoint['Prior'] = nadata_obj.Prior
    
    # 添加额外参数
    checkpoint.update(kwargs)
    
    # 保存到文件
    torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
    
    logger.info(f"检查点已保存: {filepath} (epoch {epoch})")


def export_results(nadata_obj, output_dir: str) -> None:
    """
    导出结果到指定目录
    
    Parameters:
    -----------
    nadata_obj : nadata
        要导出的nadata对象
    output_dir : str
        输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出训练结果
    train_results = nadata_obj.Model.get_train_results()
    if train_results:
        import json
        with open(os.path.join(output_dir, 'train_results.json'), 'w') as f:
            json.dump(train_results, f, indent=2, default=str)
    
    # 导出配置
    config = nadata_obj.Model.get_config()
    if config:
        import json
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    # 导出模型比较结果
    try:
        from ..model.models import compare_models
        comparison_results = compare_models(nadata_obj, verbose=0)
        if 'comparison_df' in comparison_results and comparison_results['comparison_df'] is not None:
            comparison_results['comparison_df'].to_csv(
                os.path.join(output_dir, 'model_comparison.csv'), index=False
            )
    except Exception as e:
        logger.warning(f"导出模型比较结果失败: {e}")
    
    # 导出数据摘要
    try:
        from ..model.models import get_summary
        summary = get_summary(nadata_obj)
        import json
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"导出摘要失败: {e}")
    
    logger.info(f"结果已导出到: {output_dir}") 