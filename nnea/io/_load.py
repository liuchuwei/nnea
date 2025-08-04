import os
import pandas as pd
import numpy as np
import logging
import toml
from typing import Optional, Union, Dict, Any, List
from ._nadata import nadata

# 获取logger
logger = logging.getLogger(__name__)


def CreateNNEA(config: str) -> nadata:
    """
    从不同格式的文件或文件夹中读取数据，并储存到nadata类
    
    Parameters:
    -----------
    config : str
        配置文件路径
        
    Returns:
    --------
    nadata
        包含数据的nadata对象
    """
    # 加载配置
    config_dict = load_config(config)
    
    # 创建nadata对象
    nadata_obj = nadata()
    
    # 将配置保存到Model容器
    nadata_obj.Model.set_config(config_dict)
    
    # 根据配置加载数据
    if 'dataset' in config_dict:
        dataset_config = config_dict['dataset']
        if 'path' in dataset_config:
            data_path = dataset_config['path']
            
            # 检查路径类型
            if os.path.isdir(data_path):
                # 文件夹模式
                nadata_obj = _load_from_folder(data_path, nadata_obj, dataset_config)
            elif os.path.isfile(data_path):
                # 单文件模式
                nadata_obj = _load_from_file(data_path, nadata_obj)
            else:
                raise ValueError(f"Data path does not exist: {data_path}")
    
    # 加载先验知识
    if 'nnea' in config_dict and 'piror_knowledge' in config_dict['nnea']:
        prior_config = config_dict['nnea']['piror_knowledge']
        if config_dict['nnea'].get('use_piror_knowldege', False):
            prior_path = config_dict['nnea'].get('piror_path')
            if prior_path and os.path.exists(prior_path):
                nadata_obj.Prior = load_prior(prior_path, nadata_obj.Var)
    
    return nadata_obj


def load_project(model_path: str) -> nadata:
    """
    从保存的模型文件中加载项目
    
    Parameters:
    -----------
    model_path : str
        模型文件路径
        
    Returns:
    --------
    nadata
        包含训练好的模型和数据的nadata对象
    """
    import torch
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # 加载模型状态 - 修复PyTorch 2.6兼容性问题
        try:
            # 首先尝试使用weights_only=False（兼容旧版本）
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            # 如果失败，尝试添加安全的globals
            import torch.serialization
            torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct'])
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 创建nadata对象
        nadata_obj = nadata()
        
        # 恢复核心数据
        if 'X' in checkpoint:
            nadata_obj.X = checkpoint['X']
        if 'Meta' in checkpoint:
            nadata_obj.Meta = checkpoint['Meta']
        if 'Var' in checkpoint:
            nadata_obj.Var = checkpoint['Var']
        if 'Prior' in checkpoint:
            nadata_obj.Prior = checkpoint['Prior']
        
        # 恢复配置信息到Model容器
        if 'config' in checkpoint:
            nadata_obj.Model.set_config(checkpoint['config'])
        
        # 恢复训练结果到Model容器
        if 'train_results' in checkpoint:
            nadata_obj.Model.set_train_results(checkpoint['train_results'])
        
        # 恢复数据索引到Model容器
        if 'indices' in checkpoint:
            indices = checkpoint['indices']
            nadata_obj.Model.set_indices(
                train_idx=indices.get('train'),
                test_idx=indices.get('test'),
                val_idx=indices.get('val')
            )
        
        # 恢复元数据到Model容器
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            for key, value in metadata.items():
                nadata_obj.Model.add_metadata(key, value)
        
        # 恢复模型状态（需要重新构建模型）
        if 'model_states' in checkpoint:
            model_states = checkpoint['model_states']
            # 这里需要根据配置重新构建模型，然后加载状态
            # 暂时保存模型状态，等待模型重新构建后加载
            nadata_obj.Model.add_metadata('saved_model_states', model_states)
        
        # 兼容性处理：处理旧版本的属性
        if hasattr(checkpoint, 'gene') and 'gene' in checkpoint:
            # 如果Var不存在，从gene创建Var
            if nadata_obj.Var is None:
                gene_names = checkpoint['gene']
                nadata_obj.Var = pd.DataFrame(index=gene_names, columns=['gene_name'])
                nadata_obj.Var['gene_name'] = gene_names
        
        if hasattr(checkpoint, 'sample_ids') and 'sample_ids' in checkpoint:
            # 如果Meta不存在，从sample_ids创建Meta
            if nadata_obj.Meta is None:
                sample_ids = checkpoint['sample_ids']
                nadata_obj.Meta = pd.DataFrame(index=sample_ids, columns=['sample_id'])
                nadata_obj.Meta['sample_id'] = sample_ids
        
        # 兼容性处理：处理旧版本的Model字典结构
        if 'model_state_dict' in checkpoint:
            nadata_obj.Model.add_metadata('legacy_model_state_dict', checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            nadata_obj.Model.set_train_results(checkpoint['training_history'])
        
        if 'evaluation_results' in checkpoint:
            nadata_obj.Model.add_metadata('evaluation_results', checkpoint['evaluation_results'])
        
        if 'explanation_results' in checkpoint:
            nadata_obj.Model.add_metadata('explanation_results', checkpoint['explanation_results'])
        
        if 'model_info' in checkpoint:
            nadata_obj.Model.add_metadata('model_info', checkpoint['model_info'])
        
        logger.info(f"成功加载项目: {model_path}")
        logger.info(f"数据形状: X={nadata_obj.X.shape if nadata_obj.X is not None else 'None'}")
        logger.info(f"模型数量: {len(nadata_obj.Model.models)}")
        
        return nadata_obj
        
    except Exception as e:
        logger.error(f"加载项目失败: {e}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    """
    import toml
    config = toml.load(config_path)
    logger.info(f"配置文件已加载: {config_path}")
    return config


def load_prior(prior_path: str, gene_data: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    从gmt文件中或者txt/excel/csv文件中加载先验知识
    
    Parameters:
    -----------
    prior_path : str
        先验知识文件路径
    gene_data : Optional[pd.DataFrame]
        基因数据，用于验证基因名称
        
    Returns:
    --------
    np.ndarray
        先验知识矩阵，形状为(基因集数, 基因数)
    """
    if not os.path.exists(prior_path):
        raise FileNotFoundError(f"Prior knowledge file not found: {prior_path}")
    
    if prior_path.endswith('.gmt'):
        return _load_gmt_file(prior_path, gene_data)
    elif prior_path.endswith(('.txt', '.csv', '.xlsx')):
        return _load_prior_from_table(prior_path, gene_data)
    else:
        raise ValueError(f"Unsupported prior knowledge file format: {prior_path}")


def _load_from_folder(folder_path: str, nadata_obj, dataset_config: Dict[str, Any]):
    """
    从文件夹加载数据
    
    Parameters:
    -----------
    folder_path : str
        文件夹路径
    nadata_obj : nadata
        nadata对象
    dataset_config : Dict[str, Any]
        数据集配置
        
    Returns:
    --------
    nadata
        更新后的nadata对象
    """
    # 查找表达矩阵文件
    exp_file = None
    phe_file = None
    
    # 如果配置中指定了文件名
    if 'exp_file' in dataset_config:
        exp_file = os.path.join(folder_path, dataset_config['exp_file'])
    if 'phe_file' in dataset_config:
        phe_file = os.path.join(folder_path, dataset_config['phe_file'])
    
    # 如果没有指定文件名，自动查找
    if not exp_file:
        for file in os.listdir(folder_path):
            if file.endswith(('.txt', '.csv', '.xlsx', '.h5')) and 'exp' in file.lower():
                exp_file = os.path.join(folder_path, file)
                break
    
    if not phe_file:
        for file in os.listdir(folder_path):
            if file.endswith(('.txt', '.csv', '.xlsx')) and ('phe' in file.lower() or 'meta' in file.lower()):
                phe_file = os.path.join(folder_path, file)
                break
    
    if not exp_file:
        raise ValueError(f"No expression matrix files found in {folder_path}")
    
    # 加载表达矩阵
    if exp_file.endswith('.h5'):
        import h5py
        with h5py.File(exp_file, 'r') as f:
            if 'X' in f:
                nadata_obj.X = f['X'][:]
            if 'Meta' in f:
                nadata_obj.Meta = f['Meta'][:]
            if 'Var' in f:
                nadata_obj.Var = f['Var'][:]
    else:
        # 读取表达矩阵
        if exp_file.endswith('.csv'):
            exp_data = pd.read_csv(exp_file, index_col=0)
        elif exp_file.endswith('.txt'):
            # 自动检测分隔符
            try:
                # 首先尝试逗号分隔符
                exp_data = pd.read_csv(exp_file, sep=',', index_col=0)
            except:
                try:
                    # 如果失败，尝试制表符分隔符
                    exp_data = pd.read_csv(exp_file, sep='\t', index_col=0)
                except:
                    # 最后尝试空格分隔符
                    exp_data = pd.read_csv(exp_file, sep=r'\s+', index_col=0)
        elif exp_file.endswith('.xlsx'):
            exp_data = pd.read_excel(exp_file, index_col=0)
        
        # 转置数据，使行为基因，列为样本
        if exp_data.shape[0] < exp_data.shape[1]:
            exp_data = exp_data.T
        
        nadata_obj.X = exp_data.values
        nadata_obj.Var = pd.DataFrame(index=exp_data.index, columns=['gene_name'])
        nadata_obj.Var['gene_name'] = exp_data.index.tolist()
        
        # 设置基因和样本信息
        nadata_obj.gene = exp_data.index.tolist()
        nadata_obj.sample_ids = exp_data.columns.tolist()
    
    # 加载表型数据
    if phe_file and os.path.exists(phe_file):
        if phe_file.endswith('.csv'):
            # 尝试读取，如果第一行是列名则使用，否则不使用列名
            try:
                nadata_obj.Meta = pd.read_csv(phe_file, index_col=0)
            except:
                nadata_obj.Meta = pd.read_csv(phe_file, header=None)
        elif phe_file.endswith('.txt'):
            # 自动检测分隔符，并且正确处理列名
            try:
                # 首先尝试逗号分隔符，第一行作为列名
                nadata_obj.Meta = pd.read_csv(phe_file, sep=',', header=0)
            except:
                try:
                    # 如果失败，尝试制表符分隔符，第一行作为列名
                    nadata_obj.Meta = pd.read_csv(phe_file, sep='\t', header=0)
                except:
                    try:
                        # 如果失败，尝试空格分隔符，第一行作为列名
                        nadata_obj.Meta = pd.read_csv(phe_file, sep=r'\s+', header=0)
                    except:
                        # 最后尝试不使用列名
                        nadata_obj.Meta = pd.read_csv(phe_file, sep=',', header=None)
        elif phe_file.endswith('.xlsx'):
            nadata_obj.Meta = pd.read_excel(phe_file, index_col=0)
        
        # 如果表型数据没有索引，使用样本ID
        if nadata_obj.Meta.index.name is None:
            # 确保索引长度匹配
            meta_length = len(nadata_obj.Meta)
            sample_length = len(nadata_obj.sample_ids)
            if meta_length <= sample_length:
                nadata_obj.Meta.index = nadata_obj.sample_ids[:meta_length]
            else:
                # 如果表型数据行数多于样本数，使用默认索引
                nadata_obj.Meta.index = range(meta_length)
    
    return nadata_obj


def _load_from_file(file_path: str, nadata_obj):
    """
    从单文件加载数据
    
    Parameters:
    -----------
    file_path : str
        文件路径
    nadata_obj : nadata
        nadata对象
        
    Returns:
    --------
    nadata
        更新后的nadata对象
    """
    if file_path.endswith('.h5'):
        import h5py
        with h5py.File(file_path, 'r') as f:
            if 'X' in f:
                nadata_obj.X = f['X'][:]
            if 'Meta' in f:
                nadata_obj.Meta = f['Meta'][:]
            if 'Var' in f:
                nadata_obj.Var = f['Var'][:]
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return nadata_obj


def _load_gmt_file(gmt_path: str, gene_names: Optional[List[str]] = None) -> np.ndarray:
    """
    解析.gmt文件为基因集指示矩阵
    
    Parameters:
    -----------
    gmt_path : str
        GMT文件路径
    gene_names : Optional[List[str]]
        基因名称列表，用于匹配基因集
        
    Returns:
    --------
    np.ndarray
        基因集指示矩阵，形状为 (num_genesets, num_genes)
    """
    pathways = []
    pathway_names = []
    
    with open(gmt_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) < 3:
                continue
            pathway_name = items[0]
            pathway_desc = items[1]
            pathway_genes = items[2:]
            
            pathway_names.append(pathway_name)
            pathways.append(pathway_genes)
    
    logger.info(f"从GMT文件加载了 {len(pathways)} 个基因集")
    
    # 如果没有提供基因名称列表，从基因集中收集
    if gene_names is None:
        all_genes = set()
        for pathway_genes in pathways:
            all_genes.update(pathway_genes)
        gene_names = list(all_genes)
        logger.info(f"从基因集中收集到 {len(gene_names)} 个唯一基因")
    else:
        logger.info(f"使用提供的基因列表，包含 {len(gene_names)} 个基因")
    
    # 构建基因到索引的映射
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    # 构建指示矩阵
    indicator = np.zeros((len(pathways), len(gene_names)), dtype=np.float32)
    
    # 统计匹配情况
    total_matches = 0
    geneset_matches = []
    
    for p_idx, genes in enumerate(pathways):
        matches = 0
        for gene in genes:
            if gene in gene_to_idx:
                indicator[p_idx, gene_to_idx[gene]] = 1.0
                matches += 1
        geneset_matches.append(matches)
        total_matches += matches
    
    # 记录匹配统计信息
    avg_matches = total_matches / len(pathways) if pathways else 0
    logger.info(f"基因集平均匹配基因数: {avg_matches:.1f}")
    logger.info(f"总匹配基因数: {total_matches}")
    
    # 检查是否有完全匹配的基因集
    perfect_matches = sum(1 for matches in geneset_matches if matches > 0)
    logger.info(f"有匹配基因的基因集数量: {perfect_matches}/{len(pathways)}")
    
    return indicator


def _load_prior_from_table(table_path: str, gene_names: Optional[List[str]] = None) -> np.ndarray:
    """
    从表格文件加载先验知识
    
    Parameters:
    -----------
    table_path : str
        表格文件路径
    gene_names : Optional[List[str]]
        基因名称列表，用于匹配基因集
        
    Returns:
    --------
    np.ndarray
        先验知识矩阵，形状为 (num_genesets, num_genes)
    """
    if table_path.endswith('.csv'):
        data = pd.read_csv(table_path)
    elif table_path.endswith('.txt'):
        # 自动检测分隔符
        try:
            # 首先尝试逗号分隔符
            data = pd.read_csv(table_path, sep=',')
        except:
            try:
                # 如果失败，尝试制表符分隔符
                data = pd.read_csv(table_path, sep='\t')
            except:
                # 最后尝试空格分隔符
                data = pd.read_csv(table_path, sep=r'\s+')
    elif table_path.endswith('.xlsx'):
        data = pd.read_excel(table_path)
    else:
        raise ValueError(f"Unsupported table format: {table_path}")
    
    # 假设表格有两列：pathway和gene
    if len(data.columns) < 2:
        raise ValueError("Table must have at least 2 columns: pathway and gene")
    
    pathway_col = data.columns[0]
    gene_col = data.columns[1]
    
    # 获取唯一路径和基因
    pathways = data[pathway_col].unique()
    genes = data[gene_col].unique()
    
    logger.info(f"从表格文件加载了 {len(pathways)} 个基因集")
    
    # 如果没有提供基因名称列表，使用表格中的基因
    if gene_names is None:
        gene_names = list(genes)
        logger.info(f"从表格中收集到 {len(gene_names)} 个唯一基因")
    else:
        logger.info(f"使用提供的基因列表，包含 {len(gene_names)} 个基因")
    
    # 构建基因到索引的映射
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    # 构建指示矩阵
    indicator = np.zeros((len(pathways), len(gene_names)), dtype=np.float32)
    
    # 统计匹配情况
    total_matches = 0
    geneset_matches = []
    
    for p_idx, pathway in enumerate(pathways):
        pathway_genes = data[data[pathway_col] == pathway][gene_col].tolist()
        matches = 0
        for gene in pathway_genes:
            if gene in gene_to_idx:
                indicator[p_idx, gene_to_idx[gene]] = 1.0
                matches += 1
        geneset_matches.append(matches)
        total_matches += matches
    
    # 记录匹配统计信息
    avg_matches = total_matches / len(pathways) if pathways else 0
    logger.info(f"基因集平均匹配基因数: {avg_matches:.1f}")
    logger.info(f"总匹配基因数: {total_matches}")
    
    # 检查是否有完全匹配的基因集
    perfect_matches = sum(1 for matches in geneset_matches if matches > 0)
    logger.info(f"有匹配基因的基因集数量: {perfect_matches}/{len(pathways)}")
    
    return indicator


def load_piror_knowledge(config: Dict[str, Any], gene_names: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """
    加载先验知识（基因集）
    
    Parameters:
    -----------
    config : Dict[str, Any]
        配置字典
    gene_names : Optional[List[str]]
        基因名称列表，用于匹配基因集
        
    Returns:
    --------
    Optional[np.ndarray]
        先验知识矩阵，形状为 (num_genesets, num_genes)
    """
    piror_config = config.get('nnea', {}).get('piror_knowledge', {})
    if not piror_config:
        logger.info("未配置先验知识")
        return None
    
    use_prior = piror_config.get('use_piror_knowledge', False)
    if not use_prior:
        logger.info("未启用先验知识")
        return None
    
    geneset_file = piror_config.get('piror_path')
    if not geneset_file:
        logger.warning("未指定基因集文件路径")
        return None
    
    if not os.path.exists(geneset_file):
        logger.warning(f"基因集文件不存在: {geneset_file}")
        return None
    
    try:
        # 根据文件类型加载基因集
        if geneset_file.endswith('.gmt'):
            # 加载GMT格式的基因集文件
            prior_matrix = _load_gmt_file(geneset_file, gene_names)
            logger.info(f"基因集已加载: {geneset_file}, 形状: {prior_matrix.shape}")
        elif geneset_file.endswith(('.csv', '.txt', '.xlsx')):
            # 加载表格格式的基因集文件
            prior_matrix = _load_prior_from_table(geneset_file, gene_names)
            logger.info(f"表格基因集已加载: {geneset_file}, 形状: {prior_matrix.shape}")
        else:
            logger.error(f"不支持的基因集文件格式: {geneset_file}")
            return None
        
        # 验证先验知识矩阵
        if prior_matrix is not None and prior_matrix.size > 0:
            # 计算稀疏度
            sparsity = 1.0 - np.sum(prior_matrix) / prior_matrix.size
            logger.info(f"先验知识矩阵稀疏度: {sparsity:.3f}")
            
            # 检查是否有有效的基因集，并打印基因集大小范围
            geneset_sizes = np.sum(prior_matrix, axis=1)
            valid_genesets = np.sum(geneset_sizes > 0)
            min_size = int(np.min(geneset_sizes)) if geneset_sizes.size > 0 else 0
            max_size = int(np.max(geneset_sizes)) if geneset_sizes.size > 0 else 0
            logger.info(f"基因集大小范围: 最小={min_size}, 最大={max_size}")
            logger.info(f"有效基因集数量: {valid_genesets}/{prior_matrix.shape[0]}")

            return prior_matrix
        else:
            logger.warning("先验知识矩阵为空")
            return None
            
    except Exception as e:
        logger.error(f"加载基因集失败: {e}")
        return None


def load_main_data(nadata_obj, config: Dict[str, Any]) -> None:
    """
    加载主要数据（表达矩阵和表型数据）
    """
    dataset_config = config.get('dataset', {})
    
    # 加载表达矩阵
    exp_file = dataset_config.get('expression_file')
    if not exp_file or not os.path.exists(exp_file):
        raise FileNotFoundError(f"表达矩阵文件不存在: {exp_file}")
    
    logger.info(f"正在加载表达矩阵: {exp_file}")
    
    # 根据文件类型加载表达矩阵
    if exp_file.endswith('.csv'):
        nadata_obj.X = pd.read_csv(exp_file, index_col=0)
    elif exp_file.endswith('.txt'):
        # 自动检测分隔符
        try:
            nadata_obj.X = pd.read_csv(exp_file, sep=',', index_col=0)
        except:
            try:
                nadata_obj.X = pd.read_csv(exp_file, sep='\t', index_col=0)
            except:
                nadata_obj.X = pd.read_csv(exp_file, sep=r'\s+', index_col=0)
    elif exp_file.endswith('.xlsx'):
        nadata_obj.X = pd.read_excel(exp_file, index_col=0)
    elif exp_file.endswith('.h5'):
        nadata_obj.X = pd.read_hdf(exp_file)
    else:
        raise ValueError(f"不支持的文件格式: {exp_file}")
    
    logger.info(f"表达矩阵加载完成: {nadata_obj.X.shape}")
    
    # 设置基因和样本ID
    nadata_obj.Var = pd.DataFrame(index=nadata_obj.X.index, columns=['gene_name'])
    nadata_obj.Var['gene_name'] = nadata_obj.X.index.tolist()
    
    # 确保Meta有正确的索引
    if nadata_obj.Meta is None:
        nadata_obj.Meta = pd.DataFrame(index=nadata_obj.X.columns, columns=['sample_id'])
        nadata_obj.Meta['sample_id'] = nadata_obj.X.columns.tolist()
    
    logger.info(f"基因数量: {len(nadata_obj.Var)}")
    logger.info(f"样本数量: {len(nadata_obj.Meta)}")
    
    # 加载表型数据
    phe_file = dataset_config.get('phenotype_file')
    if phe_file and os.path.exists(phe_file):
        logger.info(f"正在加载表型数据: {phe_file}")
        
        if phe_file.endswith('.csv'):
            # 尝试读取，如果第一行是列名则使用，否则不使用列名
            try:
                nadata_obj.Meta = pd.read_csv(phe_file, index_col=0)
            except:
                nadata_obj.Meta = pd.read_csv(phe_file, header=None)
        elif phe_file.endswith('.txt'):
            # 自动检测分隔符，并且正确处理列名
            try:
                # 首先尝试逗号分隔符，第一行作为列名
                nadata_obj.Meta = pd.read_csv(phe_file, sep=',', header=0)
            except:
                try:
                    # 如果失败，尝试制表符分隔符，第一行作为列名
                    nadata_obj.Meta = pd.read_csv(phe_file, sep='\t', header=0)
                except:
                    try:
                        # 如果失败，尝试空格分隔符，第一行作为列名
                        nadata_obj.Meta = pd.read_csv(phe_file, sep=r'\s+', header=0)
                    except:
                        # 最后尝试不使用列名
                        nadata_obj.Meta = pd.read_csv(phe_file, sep=',', header=None)
        elif phe_file.endswith('.xlsx'):
            nadata_obj.Meta = pd.read_excel(phe_file, index_col=0)
        
        # 如果表型数据没有索引，使用样本ID
        if nadata_obj.Meta.index.name is None:
            # 确保索引长度匹配
            meta_length = len(nadata_obj.Meta)
            sample_length = len(nadata_obj.sample_ids)
            if meta_length <= sample_length:
                nadata_obj.Meta.index = nadata_obj.sample_ids[:meta_length]
            else:
                # 如果表型数据行数多于样本数，使用默认索引
                nadata_obj.Meta.index = range(meta_length)
        
        logger.info(f"表型数据加载完成: {nadata_obj.Meta.shape}")
        logger.info(f"表型数据列: {list(nadata_obj.Meta.columns)}")
    else:
        logger.warning(f"表型数据文件不存在: {phe_file}")
        nadata_obj.Meta = None