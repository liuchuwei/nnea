"""
NNEA模型工厂
根据配置选择不同的模型类型
"""

import logging
import torch
import os
from typing import Dict, Any, Optional
from .base import BaseModel
from .nnea_classifier import NNEAClassifier
from .nnea_survival import NNEASurvival
from ..utils.helpers import ensure_reproducibility
import torch.nn as nn
import numpy as np
from .nnea_autoencoder import NNEAAutoencoder
from .nnea_regresser import NNEARegresser
from .nnea_umap import NNEAUMAP

logger = logging.getLogger(__name__)

def build_model(config: Dict[str, Any]) -> BaseModel:
    """
    根据配置构建模型
    
    Args:
        config: 模型配置
        
    Returns:
        构建好的模型实例
    """
    # 确保实验可重复性
    ensure_reproducibility(config)
    
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
        # 直接使用嵌套配置结构，不进行flatten
        model_config = config
    else:
        model_config = config
    
    if model_type == 'nnea':
        logger.info("构建NNEA分类器")
        return NNEAClassifier(model_config)
    elif model_type == 'nnea_regression':
        logger.info("构建NNEA回归器")
        return NNEARegresser(model_config)
    elif model_type == 'nnea_survival':
        logger.info("构建NNEA生存分析模型")
        return NNEASurvival(model_config)
    elif model_type == 'nnea_autoencoder':
        logger.info("构建NNEA自编码器")
        return NNEAAutoencoder(model_config)
    elif model_type == 'nnea_umap':
        logger.info("构建NNEA UMAP模型")
        return NNEAUMAP(model_config)
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
    
    # 打印模型结构信息
    if model_type == 'nnea' and hasattr(model, 'model'):
        print_model_structure(model.model)
    
    # 保存到nadata的Model容器
    nadata.Model.add_model(model_type, model)
    
    logger.info(f"模型已构建并添加到nadata.Model: {model_type}")

def print_model_structure(model):
    """
    打印NNEA模型的结构信息，特别是geneset_layer和focus_layer
    
    Args:
        model: NNEAModel实例
    """
    print("\n" + "="*60)
    print("🔍 NNEA模型结构分析")
    print("="*60)
    
    # 打印geneset_layer结构
    if hasattr(model, 'geneset_layer'):
        print("\n📊 Geneset Layer 结构:")
        print("-" * 40)
        geneset_layer = model.geneset_layer
        print(f"类型: {type(geneset_layer).__name__}")
        print(f"基因数量: {geneset_layer.num_genes}")
        print(f"基因集数量: {geneset_layer.num_sets}")
        print(f"最小基因集大小: {geneset_layer.min_set_size}")
        print(f"最大基因集大小: {geneset_layer.max_set_size}")
        print(f"先验知识: {'是' if geneset_layer.piror_knowledge is not None else '否'}")
        print(f"冻结先验: {geneset_layer.freeze_piror}")
        print(f"Dropout率: {geneset_layer.geneset_dropout.p}")
        
        # 打印基因集成员关系矩阵的形状
        if hasattr(geneset_layer, 'set_membership'):
            membership_shape = geneset_layer.set_membership.shape
            print(f"基因集成员关系矩阵形状: {membership_shape}")
            
            # 计算稀疏性
            membership = geneset_layer.set_membership.detach()
            sparsity = (membership == 0).float().mean().item()
            print(f"成员关系矩阵稀疏性: {sparsity:.3f}")
    
    # 打印focus_layer结构
    if hasattr(model, 'focus_layer'):
        print("\n🎯 Focus Layer 结构:")
        print("-" * 40)
        focus_layer = model.focus_layer
        print(f"类型: {type(focus_layer).__name__}")
        
        # 分析focus_layer的组成
        if isinstance(focus_layer, nn.Sequential):
            print(f"层数: {len(focus_layer)}")
            for i, layer in enumerate(focus_layer):
                print(f"  层 {i+1}: {type(layer).__name__}")
                if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                    print(f"    输入维度: {layer.in_features}")
                    print(f"    输出维度: {layer.out_features}")
        else:
            print(f"层结构: {focus_layer}")
    
    # 打印生物学约束层
    if hasattr(model, 'bio_constraint_layer') and model.bio_constraint_layer is not None:
        print("\n🧬 Biological Constraint Layer:")
        print("-" * 40)
        bio_layer = model.bio_constraint_layer
        print(f"类型: {type(bio_layer).__name__}")
        print(f"输入维度: {bio_layer.input_dim}")
        print(f"先验知识形状: {bio_layer.piror_knowledge.shape}")
    
    # 打印模型总体信息
    print("\n📈 模型总体信息:")
    print("-" * 40)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"设备: {next(model.parameters()).device}")
    
    print("\n" + "="*60)

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
    
    # 检查是否启用tailor策略
    config = nadata.Model.get_config()
    training_config = config.get('training', {})
    tailor_enabled = training_config.get('tailor', False)
    
    if tailor_enabled:
        # 使用tailor训练策略
        train_results = _train_with_tailor(nadata, model, verbose=verbose)
    else:
        # 使用标准训练策略
        train_results = model.train(nadata, verbose=verbose)
    
    # 保存训练结果到Model容器
    nadata.Model.set_train_results(train_results)
    
    return train_results

def _train_with_tailor(nadata, model, verbose: int = 1) -> Dict[str, Any]:
    """
    使用循环tailor策略训练模型，每过tailor_epoch个epoch都进行模型裁剪
    添加早停机制：如果连续3次tailor后验证损失没有下降，则停止训练
    
    Args:
        nadata: nadata对象
        model: 模型实例
        verbose: 详细程度
        
    Returns:
        训练结果
    """
    config = nadata.Model.get_config()
    training_config = config.get('training', {})
    
    # 获取tailor相关参数
    tailor_epoch = training_config.get('tailor_epoch', 20)
    tailor_geneset = training_config.get('tailor_geneset', 2)
    total_epochs = training_config.get('epochs', 100)
    
    # 获取输出目录（已在set_config中创建）
    outdir = config.get('global', {}).get('outdir', 'experiment/test')
    
    logger = logging.getLogger(__name__)
    logger.info(f"启用循环tailor策略: tailor_epoch={tailor_epoch}, tailor_geneset={tailor_geneset}, total_epochs={total_epochs}")
    logger.info(f"输出目录: {outdir}")
    
    # 初始化变量
    current_model = model
    current_epoch = 0
    stage_results = []
    tailor_history = []
    model_type = config.get('global', {}).get('model', 'nnea')
    
    # 早停机制变量
    best_val_loss = float('inf')
    best_model_state = None
    best_stage = 0
    no_improvement_count = 0
    max_no_improvement = 3  # 连续3次tailor后验证损失没有下降则停止
    
    # 循环训练和裁剪
    while current_epoch < total_epochs:
        # 计算当前阶段的训练轮数
        if current_epoch + tailor_epoch <= total_epochs:
            epochs_to_train = tailor_epoch
        else:
            epochs_to_train = total_epochs - current_epoch
        
        stage_num = len(stage_results) + 1
        logger.info(f"第{stage_num}阶段训练: 从第{current_epoch}个epoch训练到第{current_epoch + epochs_to_train}个epoch")
        
        # 训练当前阶段
        stage_result = current_model.train(nadata, verbose=verbose, max_epochs=epochs_to_train, continue_training=(current_epoch > 0))
        stage_results.append(stage_result)
        
        current_epoch += epochs_to_train
        
        # 获取当前阶段的验证损失
        current_val_loss = stage_result.get('final_val_loss', float('inf'))
        if current_val_loss is None:
            current_val_loss = float('inf')
        
        logger.info(f"第{stage_num}阶段验证损失: {current_val_loss:.6f}")
        
        # 检查是否为最佳模型
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_stage = stage_num
            no_improvement_count = 0
            # 保存最佳模型状态
            best_model_state = current_model.model.state_dict().copy()
            logger.info(f"✅ 第{stage_num}阶段验证损失改善到 {best_val_loss:.6f}")
        else:
            no_improvement_count += 1
            logger.info(f"⚠️ 第{stage_num}阶段验证损失未改善，连续未改善次数: {no_improvement_count}/{max_no_improvement}")
        
        # 保存当前阶段的结果
        stage_info = {
            'stage': stage_num,
            'epoch': current_epoch,
            'val_loss': current_val_loss,
            'best_val_loss': best_val_loss,
            'no_improvement_count': no_improvement_count
        }
        tailor_history.append(stage_info)
        
        logger.info(f"📊 第{stage_num}阶段训练完成，验证损失: {current_val_loss:.6f}")
        
        # 检查早停条件
        if no_improvement_count >= max_no_improvement:
            logger.info(f"🛑 连续{max_no_improvement}次tailor后验证损失未改善，触发早停！")
            logger.info(f"   最佳验证损失: {best_val_loss:.6f} (第{best_stage}阶段)")
            break
        
        # 如果还没到总轮数，进行裁剪
        if current_epoch < total_epochs:
            logger.info(f"第{stage_num}阶段训练完成，开始裁剪模型...")
            
            # 获取基因集重要性
            logger.info("计算基因集重要性...")
            try:
                explain_results = current_model.explain(nadata, method='importance')
                geneset_importance = np.array(explain_results['importance']['geneset_importance'])
                logger.info(f"基因集重要性计算完成，形状: {geneset_importance.shape}")
            except Exception as e:
                logger.error(f"基因集重要性计算失败: {e}")
                # 使用随机重要性作为备选
                geneset_importance = np.random.rand(current_model.model.num_genesets)
            
            # 确定要移除的基因集（最不重要的）
            num_genesets_to_remove = tailor_geneset
            if num_genesets_to_remove >= len(geneset_importance):
                logger.warning(f"要移除的基因集数量({num_genesets_to_remove})大于等于总基因集数量({len(geneset_importance)})，调整为移除1个")
                num_genesets_to_remove = 1
            
            # 找到最不重要的基因集索引
            least_important_indices = np.argsort(geneset_importance)[:num_genesets_to_remove]
            important_indices = np.argsort(geneset_importance)[num_genesets_to_remove:]
            
            # 尝试获取genesets_annotated的key
            genesets_annotated = nadata.uns.get('nnea_explain', {}).get('importance', {}).get('genesets', {})
            if genesets_annotated:
                # 获取genesets_annotated的key列表
                geneset_keys = list(genesets_annotated.keys())
                
                # 将索引映射到geneset key
                removed_keys = [geneset_keys[idx] if idx < len(geneset_keys) else f"Geneset_{idx}" for idx in least_important_indices]
                kept_keys = [geneset_keys[idx] if idx < len(geneset_keys) else f"Geneset_{idx}" for idx in important_indices]
                
                logger.info(f"将移除基因集: {removed_keys}")
                logger.info(f"保留基因集: {kept_keys}")
            else:
                logger.info(f"将移除基因集索引: {least_important_indices.tolist()}")
                logger.info(f"保留基因集索引: {important_indices.tolist()}")
            
            # 记录裁剪信息
            tailor_info = {
                'stage': stage_num,
                'epoch': current_epoch,
                'removed_genesets': least_important_indices.tolist(),
                'kept_genesets': important_indices.tolist(),
                'geneset_importance': geneset_importance.tolist(),
                'num_genesets_before': len(geneset_importance),
                'num_genesets_after': len(important_indices),
                'val_loss': current_val_loss,
                'no_improvement_count': no_improvement_count
            }
            
            # 如果有genesets_annotated，添加key信息
            if genesets_annotated:
                geneset_keys = list(genesets_annotated.keys())
                removed_keys = [geneset_keys[idx] if idx < len(geneset_keys) else f"Geneset_{idx}" for idx in least_important_indices]
                kept_keys = [geneset_keys[idx] if idx < len(geneset_keys) else f"Geneset_{idx}" for idx in important_indices]
                tailor_info['removed_geneset_keys'] = removed_keys
                tailor_info['kept_geneset_keys'] = kept_keys
            tailor_history.append(tailor_info)
            
            # 裁剪模型
            logger.info("开始裁剪模型...")
            cropped_model = _crop_nnea_model(nadata, current_model, important_indices, config)
            
            # 更新nadata中的模型
            nadata.Model.add_model(f"{model_type}_cropped_stage_{stage_num}", cropped_model)
            
            # 更新当前模型为裁剪后的模型
            current_model = cropped_model
            
            logger.info(f"第{stage_num}阶段裁剪完成，剩余基因集数量: {len(important_indices)}")
        else:
            logger.info("训练完成，无需进一步裁剪")
    
    # 加载最佳模型
    if best_model_state is not None:
        logger.info(f"🔄 加载最佳模型 (第{best_stage}阶段，验证损失: {best_val_loss:.6f})")
        
        # 检查当前模型与最佳模型状态的参数维度是否匹配
        current_state_dict = current_model.model.state_dict()
        best_state_dict = best_model_state
        
        # 检查关键参数维度是否匹配
        dimension_mismatch = False
        mismatch_info = []
        
        for key in best_state_dict.keys():
            if key in current_state_dict:
                if best_state_dict[key].shape != current_state_dict[key].shape:
                    dimension_mismatch = True
                    mismatch_info.append(f"{key}: {best_state_dict[key].shape} vs {current_state_dict[key].shape}")
        
        if dimension_mismatch:
            logger.warning(f"⚠️ 检测到参数维度不匹配，这可能是由于模型裁剪导致的:")
            for info in mismatch_info:
                logger.warning(f"   {info}")
            
            # 尝试从最佳模型状态重建模型
            logger.info("🔄 尝试从最佳模型状态重建模型...")
            try:
                # 从最佳模型状态推断原始配置
                best_num_genesets = best_state_dict.get('geneset_layer.query_vectors', torch.tensor([])).shape[0]
                if best_num_genesets > 0:
                    # 创建与最佳模型状态匹配的配置
                    best_config = config.copy()
                    nnea_config = best_config.get('nnea', {})
                    geneset_config = nnea_config.get('geneset_layer', {})
                    geneset_config['num_genesets'] = best_num_genesets
                    nnea_config['geneset_layer'] = geneset_config
                    best_config['nnea'] = nnea_config
                    
                    # 创建新的模型实例
                    if config.get('global').get('model') == "nnea_classifier":
                        from .nnea_classifier import NNEAClassifier
                        best_model = NNEAClassifier(best_config)
                        best_model.build(nadata)
                    elif config.get('global').get('model') == "nnea_survival":
                        from .nnea_survival import NNEASurvival
                        best_model = NNEASurvival(best_config)
                        best_model.build(nadata)
                    elif config.get('global').get('model') == "nnea_regression":
                        from .nnea_regresser import NNEARegresser
                        best_model = NNEARegresser(best_config)
                        best_model.build(nadata)

                    # 加载最佳模型状态
                    best_model.model.load_state_dict(best_model_state)
                    best_model.device = current_model.device
                    best_model.model = best_model.model.to(best_model.device)
                    
                    # 更新当前模型为最佳模型
                    current_model = best_model
                    logger.info(f"✅ 成功从最佳模型状态重建模型，基因集数量: {best_num_genesets}")
                else:
                    raise ValueError("无法从最佳模型状态推断基因集数量")
                    
            except Exception as e:
                logger.error(f"❌ 从最佳模型状态重建模型失败: {e}")
                logger.warning("⚠️ 将使用当前模型作为最终模型")
                # 保存当前模型作为最终模型
                final_model_path = os.path.join(outdir, "final_model.pth")
                torch.save(current_model.model.state_dict(), final_model_path)
                logger.info(f"💾 当前模型已保存到: {final_model_path}")
        else:
            # 参数维度匹配，直接加载
            current_model.model.load_state_dict(best_model_state)
        
        # 更新nadata中的模型为最佳模型
        nadata.Model.add_model(f"{model_type}_best", current_model)
        
        # 保存最终的最佳模型
        final_best_model_path = os.path.join(outdir, "best_model_final.pth")
        torch.save(best_model_state, final_best_model_path)
        
        final_best_nadata_path = os.path.join(outdir, "best_nadata_final.pkl")
        try:
            nadata.save(final_best_nadata_path, format="pickle", save_data=True)
            logger.info(f"💾 最终最佳模型已保存到: {final_best_model_path}")
            logger.info(f"💾 最终最佳nadata已保存到: {final_best_nadata_path}")
        except Exception as e:
            logger.error(f"保存最终最佳模型失败: {e}")
    else:
        logger.warning("⚠️ 未找到最佳模型状态，使用当前模型")
        # 保存当前模型作为最终模型
        final_model_path = os.path.join(outdir, "final_model.pth")
        torch.save(current_model.model.state_dict(), final_model_path)
        logger.info(f"💾 当前模型已保存到: {final_model_path}")
    
    # 合并训练结果
    combined_results = {
        'stage_results': stage_results,
        'tailor_history': tailor_history,
        'tailor_info': {
            'tailor_epoch': tailor_epoch,
            'tailor_geneset': tailor_geneset,
            'total_stages': len(stage_results),
            'final_geneset_count': current_model.model.num_genesets if hasattr(current_model.model, 'num_genesets') else 'unknown',
            'best_stage': best_stage,
            'best_val_loss': best_val_loss,
            'early_stopped': no_improvement_count >= max_no_improvement,
            'no_improvement_count': no_improvement_count
        }
    }
    
    logger.info(f"循环Tailor策略训练完成，共进行了{len(stage_results)}个阶段")
    if no_improvement_count >= max_no_improvement:
        logger.info(f"训练因早停而结束，最佳模型来自第{best_stage}阶段")
    else:
        logger.info(f"训练正常完成，最佳模型来自第{best_stage}阶段")
    
    return combined_results

def _crop_nnea_model(nadata, model, important_indices: np.ndarray, config: Dict[str, Any]):
    """
    裁剪NNEA模型，移除不重要的基因集
    
    Args:
        nadata: nadata对象
        model: 原始模型
        important_indices: 要保留的基因集索引
        config: 模型配置
        
    Returns:
        裁剪后的模型
    """
    logger = logging.getLogger(__name__)
    
    # 创建新的配置
    cropped_config = config.copy()
    
    # 更新基因集数量
    nnea_config = cropped_config.get('nnea', {})
    geneset_config = nnea_config.get('geneset_layer', {})
    original_num_genesets = geneset_config.get('num_genesets', 20)
    new_num_genesets = len(important_indices)
    
    geneset_config['num_genesets'] = new_num_genesets
    nnea_config['geneset_layer'] = geneset_config
    cropped_config['nnea'] = nnea_config
    
    logger.info(f"裁剪基因集数量: {original_num_genesets} -> {new_num_genesets}")
    
    # 创建新的模型实例
    if cropped_config.get('global').get("model") == 'nnea_classifier':
        from .nnea_classifier import NNEAClassifier
        cropped_model = NNEAClassifier(cropped_config)
    elif cropped_config.get('global').get('model') == 'nnea_survival':
        from .nnea_survival import NNEASurvival
        cropped_model = NNEASurvival(cropped_config)
    elif cropped_config.get('global').get('model') == 'nnea_regression':
        from .nnea_regresser import NNEARegresser
        cropped_model = NNEARegresser(cropped_config)
    # 构建新模型
    cropped_model.build(nadata)
    
    # 获取原始模型的基因集层参数
    original_geneset_params = model.model.geneset_layer.get_geneset_parameters()
    
    # 设置裁剪后模型的基因集层参数
    important_indices_tensor = torch.tensor(important_indices, dtype=torch.long)
    cropped_model.model.geneset_layer.set_geneset_parameters(original_geneset_params, important_indices_tensor)
    
    # 复制其他层的参数
    # original_state_dict = model.model.state_dict()
    # cropped_state_dict = cropped_model.model.state_dict()
    #
    # for key in original_state_dict:
    #     if key not in cropped_state_dict:
    #         continue
    #
    #     # 跳过基因集层的参数，因为已经单独处理
    #     if 'geneset_layer' in key:
    #         continue
    #     else:
    #         # 非基因集层参数直接复制
    #         cropped_state_dict[key] = original_state_dict[key]
    #
    # # 特殊处理focus_layer的第一层参数
    # # 由于geneset_layer的输出维度发生了变化，focus_layer的第一层需要相应调整
    # # _update_focus_layer_parameters(model.model, cropped_model.model, important_indices)
    #
    # # 加载裁剪后的参数
    # cropped_model.model.load_state_dict(cropped_state_dict)
    
    # 设置设备
    cropped_model.device = model.device
    cropped_model.model = cropped_model.model.to(cropped_model.device)
    
    logger.info("模型裁剪完成")
    return cropped_model


def _update_focus_layer_parameters(original_model, cropped_model, important_indices: np.ndarray):
    """
    更新focus_layer的参数，以适应geneset_layer输出维度的变化
    
    Args:
        original_model: 原始模型
        cropped_model: 裁剪后的模型
        important_indices: 要保留的基因集索引
    """
    logger = logging.getLogger(__name__)
    
    # 获取原始和裁剪后的focus_layer
    original_focus_layer = original_model.focus_layer
    cropped_focus_layer = cropped_model.focus_layer
    
    if not isinstance(original_focus_layer, nn.Sequential) or not isinstance(cropped_focus_layer, nn.Sequential):
        logger.warning("focus_layer不是Sequential结构，跳过参数更新")
        return
    
    # 找到第一个Linear层（通常是focus_layer的第一层）
    first_linear_layer = None
    first_linear_idx = None
    
    for i, layer in enumerate(original_focus_layer):
        if isinstance(layer, nn.Linear):
            first_linear_layer = layer
            first_linear_idx = i
            break
    
    if first_linear_layer is None:
        logger.warning("未找到Linear层，跳过focus_layer参数更新")
        return
    
    # 获取裁剪后模型的第一层Linear层
    cropped_first_linear = cropped_focus_layer[first_linear_idx]
    
    if not isinstance(cropped_first_linear, nn.Linear):
        logger.warning("裁剪后模型的第一层不是Linear层，跳过参数更新")
        return
    
    # 检查维度是否匹配
    original_input_dim = first_linear_layer.in_features
    cropped_input_dim = cropped_first_linear.in_features
    
    if original_input_dim != len(important_indices):
        logger.warning(f"维度不匹配: 原始输入维度 {original_input_dim}, 重要索引数量 {len(important_indices)}")
        return
    
    # 更新权重：只保留对应重要基因集的权重
    original_weight = first_linear_layer.weight.data
    original_bias = first_linear_layer.bias.data if first_linear_layer.bias is not None else None
    
    # 选择对应重要基因集的权重行
    important_indices_tensor = torch.tensor(important_indices, dtype=torch.long, device=original_weight.device)
    cropped_weight = original_weight[:, important_indices_tensor]
    
    # 更新裁剪后模型的权重
    cropped_first_linear.weight.data = cropped_weight
    
    # 偏置项保持不变（如果有的话）
    if original_bias is not None and cropped_first_linear.bias is not None:
        cropped_first_linear.bias.data = original_bias.clone()
    
    logger.info(f"focus_layer第一层参数更新完成: 输入维度 {original_input_dim} -> {cropped_input_dim}")

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

def predict(nadata, split='test', model_name: Optional[str] = None, return_probabilities: bool = True) -> Dict[str, Any]:
    """
    模型预测

    Args:
        nadata: nadata对象
        split: 预测的数据集分割
        model_name: 模型名称
        return_probabilities: 是否返回概率值

    Returns:
        预测结果字典，包含预测值、概率值、真实标签等
    """
    from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
    import torch

    logger = logging.getLogger(__name__)

    if not nadata.Model.models:
        raise ValueError("nadata.Model中没有模型，请先调用build()")

    # 确定要预测的模型
    if model_name is None:
        model_type = nadata.Model.get_config().get('global', {}).get('model', 'nnea')
        model = nadata.Model.get_model(model_type)
    else:
        model = nadata.Model.get_model(model_name)

    if model is None:
        raise ValueError(f"未找到模型: {model_name or 'default'}")

    # 获取数据索引
    indices = nadata.Model.get_indices(split)
    if indices is None:
        logger.warning(f"未找到{split}集的索引，无法进行预测")
        return {
            'y_test': None,
            'y_pred': None,
            'y_proba': None,
            'predictions': None,
            'error': f"未找到{split}集的索引"
        }

    try:
        # 获取测试集数据
        X_test = nadata.X[indices]  # 转置为(样本数, 特征数)

        # 获取目标列名
        config = nadata.Model.get_config()
        target_col = config.get('dataset', {}).get('target_column', 'target')
        y_test = nadata.Meta.iloc[indices][target_col].values

        logger.info(f"🔮 进行模型预测...")
        logger.info(f"📊 测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

        # 模型预测
        model.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(model.device)
            outputs = model.model(X_test_tensor)

        # 根据任务类型处理预测结果
        task_type = getattr(model, 'task', 'classification')
        
        if task_type == 'classification':
            # 分类任务处理
            if outputs.shape[1] == 2:
                # 二分类情况
                y_proba = torch.softmax(outputs, dim=1).cpu().numpy()[:, 1]
                y_pred = (y_proba > 0.5).astype(int)
            else:
                # 多分类情况
                y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                y_pred = np.argmax(y_proba, axis=1)

            # 计算评估指标
            if len(np.unique(y_test)) == 2:
                # 二分类
                auc = roc_auc_score(y_test, y_proba)
                logger.info(f"📊 测试集AUC：{auc:.4f}")
            else:
                # 多分类
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                logger.info(f"📊 测试集AUC：{auc:.4f}")

            # 输出分类报告
            logger.info("📊 预测结果:")
            logger.info(f"测试集分类报告：\n{classification_report(y_test, y_pred)}")

            # 保存预测结果到Model容器
            prediction_results = {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'predictions': outputs.cpu().numpy(),
                'auc': auc,
                'split': split,
                'task_type': 'classification'
            }

        elif task_type == 'survival':
            # 生存任务处理
            time_col = config.get('dataset', {}).get('time_column', 'Time')
            event_col = config.get('dataset', {}).get('event_column', 'Event')
            
            times = nadata.Meta.iloc[indices][time_col].values
            events = nadata.Meta.iloc[indices][event_col].values
            
            # 生存分析预测结果
            risk_scores = outputs.cpu().numpy().flatten()
            
            # 计算生存分析指标
            from lifelines.utils import concordance_index
            c_index = concordance_index(times, -risk_scores, events)
            
            logger.info(f"📊 测试集C-index：{c_index:.4f}")
            
            # 保存预测结果到Model容器
            prediction_results = {
                'times': times,
                'events': events,
                'risk_scores': risk_scores,
                'predictions': outputs.cpu().numpy(),
                'c_index': c_index,
                'split': split,
                'task_type': 'survival'
            }

        elif task_type == 'regression':
            # 回归任务处理
            predictions = outputs.cpu().numpy().flatten()
            
            # 计算回归评估指标
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mse)
            
            logger.info(f"📊 测试集回归指标：")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            
            # 保存预测结果到Model容器
            prediction_results = {
                'y_test': y_test,
                'predictions': predictions,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'split': split,
                'task_type': 'regression'
            }

        else:
            # 其他任务类型
            predictions = outputs.cpu().numpy()
            prediction_results = {
                'y_test': y_test,
                'predictions': predictions,
                'split': split,
                'task_type': task_type
            }

        # 保存到nadata的Model容器
        nadata.Model.add_metadata('prediction_results', prediction_results)

        logger.info(f"✅ 模型预测完成，结果已保存到nadata.Model")

        return prediction_results

    except Exception as e:
        logger.error(f"❌ 模型预测失败: {e}")
        return {
            'y_test': None,
            'y_pred': None,
            'y_proba': None,
            'predictions': None,
            'error': str(e)
        }