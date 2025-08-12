#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单细胞测序数据处理脚本
用于构建pbmc3k和ifnb的nadata类，包含uns字典用于存储PCA等数据
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import pickle
from scipy import sparse
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 直接导入nadata类，避免触发整个nnea包的导入
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnea.io._nadata import nadata


class SingleCellProcessor:
    """单细胞数据处理类"""
    
    def __init__(self):
        self.datasets_dir = "../../datasets"
        
    def load_h5_data(self, h5_path: str) -> Dict[str, Any]:
        """
        从HDF5文件加载单细胞数据
        
        Parameters:
        -----------
        h5_path : str
            HDF5文件路径
            
        Returns:
        --------
        dict : 包含所有数据的字典
        """
        print(f"正在加载数据: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            # 获取稀疏矩阵参数
            i = f['i'][:]
            p = f['p'][:]
            x = f['x'][:]
            dims = f['dims'][:]
            rownames = f['rownames'][:] if 'rownames' in f else None
            colnames = f['colnames'][:] if 'colnames' in f else None
            
            # 重建稀疏矩阵
            if rownames is not None:
                rownames = [name.decode('utf-8') if isinstance(name, bytes) else name 
                           for name in rownames]
            if colnames is not None:
                colnames = [name.decode('utf-8') if isinstance(name, bytes) else name 
                           for name in colnames]
            
            # 创建稀疏矩阵
            X = sparse.csc_matrix((x, i, p), shape=tuple(dims))
            if rownames is not None:
                X = pd.DataFrame(X.toarray(), index=rownames, columns=colnames)
            else:
                X = pd.DataFrame(X.toarray())
            
            # 加载元数据
            # 由于R写入HDF5时Meta通常是按列优先（Fortran顺序）存储，numpy读取时会导致行列颠倒，所以需要转置
            Meta = f['Meta'][:].T  # 恢复原始行列
            # 恢复字符串（去除b""前缀）
            meta_columns = [col.decode('utf-8') if isinstance(col, bytes) else col for col in f['Meta_columns'][:]]
            # 检查是否有字符串列需要解码
            Meta_df = pd.DataFrame(Meta, columns=meta_columns)
            for col in Meta_df.columns:
                # 处理字符串类型的列（包括dtype为object和numpy字节串类型，如dtype('S12')）
                Meta_df[col] = Meta_df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            Meta = Meta_df
            Meta = pd.DataFrame(Meta, columns=meta_columns)
            
            # 加载PCA数据
            pca = f['pca'][:].T if 'pca' in f else None
            
        return {
            'X': X,
            'Meta': Meta,
            'pca': pca,
            'rownames': rownames,
            'colnames': colnames
        }
    
    def create_uns_dict(self, data_dict: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """
        创建uns字典，存储单细胞数据的额外信息
        
        Parameters:
        -----------
        data_dict : dict
            包含数据的字典
        dataset_name : str
            数据集名称
            
        Returns:
        --------
        dict : uns字典
        """
        uns = {}
        
        # 存储PCA数据
        if data_dict['pca'] is not None:
            uns['pca'] = data_dict['pca']
            uns['pca_components'] = data_dict['pca'].shape[1]
        
        # 存储数据集信息
        uns['dataset_name'] = dataset_name
        uns['n_cells'] = data_dict['X'].shape[1]
        uns['n_genes'] = data_dict['X'].shape[0]
        
        # 存储细胞类型信息
        if 'cell_types' in data_dict['Meta'].columns:
            cell_types = data_dict['Meta']['cell_types'].value_counts()
            uns['cell_type_counts'] = cell_types.to_dict()
            uns['n_cell_types'] = len(cell_types)
        
        # 存储实验条件信息（针对IFNB数据集）
        if 'stim' in data_dict['Meta'].columns:
            stim_conditions = data_dict['Meta']['stim'].value_counts()
            uns['stim_conditions'] = stim_conditions.to_dict()
            uns['n_conditions'] = len(stim_conditions)
        
        # 存储基因和细胞名称
        if data_dict['rownames'] is not None:
            uns['gene_names'] = data_dict['rownames']
        if data_dict['colnames'] is not None:
            uns['cell_names'] = data_dict['colnames']
        
        return uns
    
    def process_pbmc3k(self):
        """处理PBMC3K数据集"""
        print("\n" + "="*50)
        print("开始处理 PBMC3K 数据集")
        print("="*50)
        
        # 文件路径
        h5_path = os.path.join(self.datasets_dir, "sc_pbmc3k", "sc_pbmc3k.h5")
        output_path = os.path.join(self.datasets_dir, "sc_pbmc3k", "pbmc3k_na.pkl")
        
        # 检查文件是否存在
        if not os.path.exists(h5_path):
            print(f"错误: 找不到文件 {h5_path}")
            return None
        
        # 加载数据
        data_dict = self.load_h5_data(h5_path)
        
        # 创建uns字典
        uns = self.create_uns_dict(data_dict, "pbmc3k")
        
        # 创建nadata对象
        na_data = nadata(
            X=data_dict['X'],
            Meta=data_dict['Meta'],
            uns=uns
        )
        
        # 保存数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(na_data, f)
        
        print(f"PBMC3K 数据处理完成！")
        print(f"- 细胞数: {uns['n_cells']}")
        print(f"- 基因数: {uns['n_genes']}")
        print(f"- 细胞类型数: {uns.get('n_cell_types', 'N/A')}")
        print(f"- 保存路径: {output_path}")
        
        return na_data
    
    def process_ifnb(self):
        """处理IFNB数据集"""
        print("\n" + "="*50)
        print("开始处理 IFNB 数据集")
        print("="*50)
        
        # 文件路径
        h5_path = os.path.join(self.datasets_dir, "sc_ifnb", "sc_ifnb.h5")
        output_path = os.path.join(self.datasets_dir, "sc_ifnb", "ifnb_na.pkl")
        
        # 检查文件是否存在
        if not os.path.exists(h5_path):
            print(f"错误: 找不到文件 {h5_path}")
            return None
        
        # 加载数据
        data_dict = self.load_h5_data(h5_path)
        
        # 创建uns字典
        uns = self.create_uns_dict(data_dict, "ifnb")
        
        # 创建nadata对象
        na_data = nadata(
            X=data_dict['X'],
            Meta=data_dict['Meta']
        )
        
        # 添加uns属性
        na_data.uns = uns
        
        # 保存数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(na_data, f)
        
        print(f"IFNB 数据处理完成！")
        print(f"- 细胞数: {uns['n_cells']}")
        print(f"- 基因数: {uns['n_genes']}")
        print(f"- 细胞类型数: {uns.get('n_cell_types', 'N/A')}")
        print(f"- 实验条件数: {uns.get('n_conditions', 'N/A')}")
        print(f"- 保存路径: {output_path}")
        
        return na_data
    
    def validate_data(self, na_data: nadata, dataset_name: str):
        """
        验证nadata对象的完整性
        
        Parameters:
        -----------
        na_data : nadata
            nadata对象
        dataset_name : str
            数据集名称
        """
        print(f"\n验证 {dataset_name} 数据完整性:")
        
        # 检查基本属性
        print(f"- X shape: {na_data.X.shape if na_data.X is not None else 'None'}")
        print(f"- Meta shape: {na_data.Meta.shape if na_data.Meta is not None else 'None'}")
        print(f"- uns keys: {list(na_data.uns.keys()) if hasattr(na_data, 'uns') else 'No uns dict'}")
        
        # 检查uns字典内容
        if hasattr(na_data, 'uns'):
            uns = na_data.uns
            print(f"- 数据集名称: {uns.get('dataset_name', 'N/A')}")
            print(f"- 细胞数: {uns.get('n_cells', 'N/A')}")
            print(f"- 基因数: {uns.get('n_genes', 'N/A')}")
            
            if 'cell_type_counts' in uns:
                print(f"- 细胞类型分布:")
                for cell_type, count in list(uns['cell_type_counts'].items())[:5]:  # 只显示前5个
                    print(f"  {cell_type}: {count}")
            
            if 'stim_conditions' in uns:
                print(f"- 实验条件分布:")
                for condition, count in uns['stim_conditions'].items():
                    print(f"  {condition}: {count}")
        
        print(f"{dataset_name} 数据验证完成！")
    
    def run(self):
        """运行完整的数据处理流程"""
        print("开始单细胞数据处理...")
        
        # 处理PBMC3K数据
        pbmc3k_data = self.process_pbmc3k()
        if pbmc3k_data is not None:
            self.validate_data(pbmc3k_data, "PBMC3K")
        
        # 处理IFNB数据
        ifnb_data = self.process_ifnb()
        if ifnb_data is not None:
            self.validate_data(ifnb_data, "IFNB")
        
        print("\n" + "="*50)
        print("所有数据处理完成！")
        print("="*50)
        
        # 显示保存的文件
        pbmc3k_path = os.path.join(self.datasets_dir, "sc_pbmc3k", "pbmc3k_na.pkl")
        ifnb_path = os.path.join(self.datasets_dir, "sc_ifnb", "ifnb_na.pkl")
        
        if os.path.exists(pbmc3k_path):
            size = os.path.getsize(pbmc3k_path) / (1024*1024)  # MB
            print(f"- PBMC3K: {pbmc3k_path} ({size:.2f} MB)")
        
        if os.path.exists(ifnb_path):
            size = os.path.getsize(ifnb_path) / (1024*1024)  # MB
            print(f"- IFNB: {ifnb_path} ({size:.2f} MB)")


def main():
    """主函数"""
    processor = SingleCellProcessor()
    processor.run()


if __name__ == "__main__":
    main()
