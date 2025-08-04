"""
基因集富集分析模块
实现类似clusterProfiler的enricher功能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import hypergeom
import logging

logger = logging.getLogger(__name__)


def load_gmt_file(gmt_path: str) -> Dict[str, List[str]]:
    """
    加载GMT格式的基因集文件
    
    Parameters:
    -----------
    gmt_path : str
        GMT文件路径
        
    Returns:
    --------
    Dict[str, List[str]]
        基因集字典，键为基因集名称，值为基因列表
    """
    genesets = {}
    
    try:
        with open(gmt_path, 'r', encoding='utf-8') as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) < 3:
                    continue
                pathway_name = items[0]
                pathway_desc = items[1]
                pathway_genes = items[2:]
                
                genesets[pathway_name] = pathway_genes
        
        logger.info(f"成功加载 {len(genesets)} 个基因集")
        return genesets
        
    except Exception as e:
        logger.error(f"加载GMT文件失败: {e}")
        return {}


def enricher(gene: List[str], 
             TERM2GENE: Dict[str, List[str]], 
             pvalueCutoff: float = 0.05,
             minGSSize: int = 10,
             maxGSSize: int = 500) -> pd.DataFrame:
    """
    基因集富集分析，类似clusterProfiler的enricher函数
    
    Parameters:
    -----------
    gene : List[str]
        待分析的基因列表
    TERM2GENE : Dict[str, List[str]]
        基因集字典，键为基因集名称，值为基因列表
    pvalueCutoff : float
        p值阈值，默认0.05
    minGSSize : int
        最小基因集大小，默认10
    maxGSSize : int
        最大基因集大小，默认500
        
    Returns:
    --------
    pd.DataFrame
        富集分析结果，包含以下列：
        - ID: 基因集ID
        - Description: 基因集描述
        - GeneRatio: 基因比例
        - BgRatio: 背景比例
        - pvalue: p值
        - p.adjust: 校正后的p值
        - qvalue: q值
        - geneID: 富集的基因ID
        - Count: 富集基因数量
    """
    if not gene or not TERM2GENE:
        logger.warning("基因列表或基因集为空")
        return pd.DataFrame()
    
    # 获取背景基因集（所有基因集的并集）
    background_genes = set()
    for genes in TERM2GENE.values():
        background_genes.update(genes)
    background_genes = list(background_genes)
    
    # 过滤基因集大小
    filtered_TERM2GENE = {}
    for term, genes in TERM2GENE.items():
        if minGSSize <= len(genes) <= maxGSSize:
            filtered_TERM2GENE[term] = genes
    
    # logger.info(f"过滤后基因集数量: {len(filtered_TERM2GENE)}")
    
    # 进行超几何检验
    results = []
    
    for term, genes in filtered_TERM2GENE.items():
        # 计算交集
        intersection = set(gene) & set(genes)
        k = len(intersection)  # 交集大小
        
        if k == 0:
            continue
        
        # 超几何检验参数
        N = len(background_genes)  # 背景基因总数
        K = len(genes)  # 基因集大小
        n = len(gene)  # 输入基因数量
        
        # 计算p值
        pvalue = hypergeom.sf(k-1, N, K, n)
        
        # 计算基因比例
        gene_ratio = f"{k}/{n}"
        bg_ratio = f"{K}/{N}"
        
        # 富集的基因ID
        geneID = "/".join(intersection)
        
        results.append({
            'ID': term,
            'Description': term,  # 可以添加描述信息
            'GeneRatio': gene_ratio,
            'BgRatio': bg_ratio,
            'pvalue': pvalue,
            'p.adjust': pvalue,  # 暂时不进行多重检验校正
            'qvalue': pvalue,    # 暂时不进行FDR校正
            'geneID': geneID,
            'Count': k
        })
    
    if not results:
        logger.warning("没有找到显著富集的基因集")
        return pd.DataFrame()
    
    # 转换为DataFrame
    result_df = pd.DataFrame(results)
    
    # 按p值排序
    result_df = result_df.sort_values('pvalue')
    
    # # 过滤p值
    # filter_result_df = result_df[result_df['pvalue'] <= pvalueCutoff]
    #
    # # logger.info(f"找到 {len(filter_result_df)} 个显著富集的基因集")
    #
    return result_df


def refine_genesets(geneset_assignments: np.ndarray,
                   geneset_importance: np.ndarray,
                   gene_names: List[str],
                   min_set_size: int,
                   max_set_size: int,
                   geneset_threshold: float = 1e-5) -> List[List[str]]:
    """
    根据基因集分配和重要性，精炼基因集
    使用TrainableGeneSetLayer中的geneset_threshold作为阈值
    
    Parameters:
    -----------
    geneset_assignments : np.ndarray
        基因集分配矩阵，形状为 (num_genesets, num_genes)
    geneset_importance : np.ndarray
        基因集重要性分数，形状为 (num_genesets,)
    gene_names : List[str]
        基因名称列表
    min_set_size : int
        最小基因集大小
    max_set_size : int
        最大基因集大小
    geneset_threshold : float
        基因集阈值，来自TrainableGeneSetLayer
        
    Returns:
    --------
    List[List[str]]
        精炼后的基因集列表
    """
    if geneset_assignments.size == 0:
        return []
    
    refined_genesets = []
    
    # 使用固定的geneset_threshold
    for j in range(geneset_assignments.shape[0]):
        gene_assignments = geneset_assignments[j]
        
        # 使用geneset_threshold选择基因
        selected_indices = np.where(gene_assignments >= geneset_threshold)[0]
        selected_genes_list = [gene_names[idx] for idx in selected_indices]
        
        # 确保基因集大小在指定范围内
        if len(selected_genes_list) > max_set_size:
            # 如果超过最大大小，按重要性排序取前max_set_size个
            gene_importance_selected = gene_assignments[selected_indices]
            sorted_indices = np.argsort(gene_importance_selected)[::-1]
            selected_indices = selected_indices[sorted_indices[:max_set_size]]
            selected_genes_list = [gene_names[idx] for idx in selected_indices]
        
        # 只保留满足最小大小要求的基因集
        if len(selected_genes_list) >= min_set_size:
            refined_genesets.append(selected_genes_list)
            logger.debug(f"基因集 {j+1}: {len(selected_genes_list)} 个基因")
        else:
            logger.warning(f"基因集 {j+1} 大小 ({len(selected_genes_list)}) 小于最小要求 ({min_set_size})")
    
    logger.info(f"使用阈值 {geneset_threshold:.6f} 精炼后基因集数量: {len(refined_genesets)}")
    return refined_genesets


def annotate_genesets(genesets: List[List[str]], 
                     gmt_path: str,
                     pvalueCutoff: float = 0.05) -> Dict[str, Any]:
    """
    对基因集进行注释
    
    Parameters:
    -----------
    genesets : List[List[str]]
        基因集列表
    gmt_path : str
        GMT文件路径
    pvalueCutoff : float
        p值阈值
        
    Returns:
    --------
    Dict[str, Any]
        注释结果，使用enrichment DataFrame的ID列作为key
    """
    # 加载基因集数据库
    term2gene = load_gmt_file(gmt_path)
    
    if not term2gene:
        logger.warning("无法加载基因集数据库")
        return {}
    
    annotation_results = {}
    
    for i, geneset in enumerate(genesets):
        if not geneset:
            continue
            
        # 进行富集分析
        enrich_result = enricher(
            gene=geneset,
            TERM2GENE=term2gene,
            pvalueCutoff=pvalueCutoff
        )
        
        # 使用enrichment DataFrame的ID列作为key
        if not enrich_result.empty:
            for _, row in enrich_result.iterrows():
                pathway_id = row['ID']
                annotation_results[pathway_id] = {
                    'genes': geneset,
                    'enrichment': enrich_result,
                    'geneset_index': i + 1
                }
        else:
            # 如果没有富集结果，使用默认key
            annotation_results[f"geneset_{i+1}"] = {
                'genes': geneset,
                'enrichment': enrich_result,
                'geneset_index': i + 1
            }
    
    logger.info(f"完成 {len(annotation_results)} 个基因集的注释")
    return annotation_results