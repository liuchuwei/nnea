"""
Gene Set Enrichment Analysis Module
Implements functionality similar to clusterProfiler's enricher
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import hypergeom
import logging

logger = logging.getLogger(__name__)


def load_gmt_file(gmt_path: str) -> Dict[str, List[str]]:
    """
    Load GMT format gene set file
    
    Parameters:
    -----------
    gmt_path : str
        GMT file path
        
    Returns:
    --------
    Dict[str, List[str]]
        Gene set dictionary, keys are gene set names, values are gene lists
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
        
        logger.info(f"Successfully loaded {len(genesets)} gene sets")
        return genesets
        
    except Exception as e:
        logger.error(f"Failed to load GMT file: {e}")
        return {}


def enricher(gene: List[str], 
             TERM2GENE: Dict[str, List[str]], 
             pvalueCutoff: float = 0.05,
             minGSSize: int = 10,
             maxGSSize: int = 500) -> pd.DataFrame:
    """
    Gene set enrichment analysis, similar to clusterProfiler's enricher function
    
    Parameters:
    -----------
    gene : List[str]
        List of genes to analyze
    TERM2GENE : Dict[str, List[str]]
        Gene set dictionary, keys are gene set names, values are gene lists
    pvalueCutoff : float
        p-value threshold, default 0.05
    minGSSize : int
        Minimum gene set size, default 10
    maxGSSize : int
        Maximum gene set size, default 500
        
    Returns:
    --------
    pd.DataFrame
        Enrichment analysis results, containing the following columns:
        - ID: Gene set ID
        - Description: Gene set description
        - GeneRatio: Gene ratio
        - BgRatio: Background ratio
        - pvalue: p-value
        - p.adjust: Adjusted p-value
        - qvalue: q-value
        - geneID: Enriched gene IDs
        - Count: Number of enriched genes
    """
    if not gene or not TERM2GENE:
        logger.warning("Gene list or gene sets are empty")
        return pd.DataFrame()
    
    # Get background gene set (union of all gene sets)
    background_genes = set()
    for genes in TERM2GENE.values():
        background_genes.update(genes)
    background_genes = list(background_genes)
    
    # Filter gene set sizes
    filtered_TERM2GENE = {}
    for term, genes in TERM2GENE.items():
        if minGSSize <= len(genes) <= maxGSSize:
            filtered_TERM2GENE[term] = genes
    
    # logger.info(f"Number of gene sets after filtering: {len(filtered_TERM2GENE)}")
    
    # Perform hypergeometric test
    results = []
    
    for term, genes in filtered_TERM2GENE.items():
        # Calculate intersection
        intersection = set(gene) & set(genes)
        k = len(intersection)  # Intersection size
        
        if k == 0:
            continue
        
        # Hypergeometric test parameters
        N = len(background_genes)  # Total background genes
        K = len(genes)  # Gene set size
        n = len(gene)  # Input gene count
        
        # Calculate p-value
        pvalue = hypergeom.sf(k-1, N, K, n)
        
        # Calculate gene ratios
        gene_ratio = f"{k}/{n}"
        bg_ratio = f"{K}/{N}"
        
        # Enriched gene IDs
        geneID = "/".join(intersection)
        
        results.append({
            'ID': term,
            'Description': term,  # Can add description information
            'GeneRatio': gene_ratio,
            'BgRatio': bg_ratio,
            'pvalue': pvalue,
            'p.adjust': pvalue,  # Temporarily no multiple testing correction
            'qvalue': pvalue,    # Temporarily no FDR correction
            'geneID': geneID,
            'Count': k
        })
    
    if not results:
        logger.warning("No significantly enriched gene sets found")
        return pd.DataFrame()
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Sort by p-value
    result_df = result_df.sort_values('pvalue')
    
    # # Filter p-values
    # filter_result_df = result_df[result_df['pvalue'] <= pvalueCutoff]
    #
    # # logger.info(f"Found {len(filter_result_df)} significantly enriched gene sets")
    #
    return result_df


def refine_genesets(geneset_assignments: np.ndarray,
                   geneset_importance: np.ndarray,
                   gene_names: List[str],
                   min_set_size: int,
                   max_set_size: int,
                   geneset_threshold: float = 1e-5) -> List[List[str]]:
    """
    Refine gene sets based on gene set assignments and importance
    Use geneset_threshold from TrainableGeneSetLayer as threshold
    
    Parameters:
    -----------
    geneset_assignments : np.ndarray
        Gene set assignment matrix, shape (num_genesets, num_genes)
    geneset_importance : np.ndarray
        Gene set importance scores, shape (num_genesets,)
    gene_names : List[str]
        List of gene names
    min_set_size : int
        Minimum gene set size
    max_set_size : int
        Maximum gene set size
    geneset_threshold : float
        Gene set threshold from TrainableGeneSetLayer
        
    Returns:
    --------
    List[List[str]]
        Refined gene set list
    """
    if geneset_assignments.size == 0:
        return []
    
    refined_genesets = []
    
    # Use fixed geneset_threshold
    for j in range(geneset_assignments.shape[0]):
        gene_assignments = geneset_assignments[j]
        
        # Use geneset_threshold to select genes
        selected_indices = np.where(gene_assignments >= geneset_threshold)[0]
        selected_genes_list = [gene_names[idx] for idx in selected_indices]
        
        # Ensure gene set size is within specified range
        if len(selected_genes_list) > max_set_size:
            # If exceeding maximum size, sort by importance and take top max_set_size
            gene_importance_selected = gene_assignments[selected_indices]
            sorted_indices = np.argsort(gene_importance_selected)[::-1]
            selected_indices = selected_indices[sorted_indices[:max_set_size]]
            selected_genes_list = [gene_names[idx] for idx in selected_indices]
        
        # Only keep gene sets that meet minimum size requirement
        if len(selected_genes_list) >= min_set_size:
            refined_genesets.append(selected_genes_list)
            logger.debug(f"Gene set {j+1}: {len(selected_genes_list)} genes")
        else:
            logger.warning(f"Gene set {j+1} size ({len(selected_genes_list)}) is less than minimum requirement ({min_set_size})")
    
    logger.info(f"Number of gene sets after refinement with threshold {geneset_threshold:.6f}: {len(refined_genesets)}")
    return refined_genesets


def annotate_genesets(genesets: List[List[str]], 
                     gmt_path: str,
                     pvalueCutoff: float = 0.05) -> Dict[str, Any]:
    """
    Annotate gene sets
    
    Parameters:
    -----------
    genesets : List[List[str]]
        List of gene sets
    gmt_path : str
        GMT file path
    pvalueCutoff : float
        p-value threshold
        
    Returns:
    --------
    Dict[str, Any]
        Annotation results, using enrichment DataFrame's ID column as key
    """
    # Load gene set database
    term2gene = load_gmt_file(gmt_path)
    
    if not term2gene:
        logger.warning("Unable to load gene set database")
        return {}
    
    annotation_results = {}
    
    for i, geneset in enumerate(genesets):
        if not geneset:
            continue
            
        # Perform enrichment analysis
        enrich_result = enricher(
            gene=geneset,
            TERM2GENE=term2gene,
            pvalueCutoff=pvalueCutoff
        )
        
        # Use enrichment DataFrame's ID column as key
        annotation_results[f"geneset_{i+1}"] = {
            'genes': geneset,
            'enrichment': enrich_result,
            'geneset_index': i + 1
        }
    
    logger.info(f"Completed annotation of {len(annotation_results)} gene sets")
    return annotation_results