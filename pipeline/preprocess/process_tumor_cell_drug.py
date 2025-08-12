#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肿瘤和细胞系药物数据交集对齐处理脚本
将datasets/tumor_drug和datasets/cell_drug中common的nadata project保留，
基因特征进行取交集对齐，创建对齐后的数据集
"""

import warnings
import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
import pickle
import re
import glob

# 忽略pandas的pyarrow警告
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='pyarrow')

# 添加nnea模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from nnea.io._nadata import nadata
except ImportError as e:
    print(f"导入nnea模块失败: {e}")
    print("请确保nnea模块已正确安装")
    sys.exit(1)

class TumorCellDrugAligner:
    """肿瘤和细胞系药物数据对齐器"""
    
    def __init__(self):
        """
        初始化对齐器
        """
        self.results = {}
        
        # 数据文件路径 - 使用绝对路径
        base_dir = Path(__file__).parent
        self.tumor_drug_dir = base_dir / "datasets" / "tumor_drug"
        self.cell_drug_dir = base_dir / "datasets" / "cell_drug"
        
        # 检查目录是否存在
        self._check_directories()
        
        # 获取药物列表
        self._get_drug_lists()
    
    def _check_directories(self):
        """检查必要目录是否存在"""
        print("检查数据目录...")
        
        if not self.tumor_drug_dir.exists():
            print(f"警告: 肿瘤药物数据目录不存在: {self.tumor_drug_dir}")
            print("请先运行肿瘤药物数据处理脚本")
            return
        
        if not self.cell_drug_dir.exists():
            print(f"警告: 细胞系药物数据目录不存在: {self.cell_drug_dir}")
            print("请先运行细胞系药物数据处理脚本")
            return
        
        print("数据目录检查完成")
    
    def _get_drug_lists(self):
        """获取肿瘤和细胞系药物列表"""
        print("获取药物列表...")
        
        # 获取肿瘤药物列表
        tumor_drugs = set()
        if self.tumor_drug_dir.exists():
            pkl_files = [f for f in self.tumor_drug_dir.iterdir() if f.name.endswith('_drug.pkl')]
            for pkl_file in pkl_files:
                drug_name = pkl_file.stem.replace('_drug', '')
                tumor_drugs.add(drug_name)
        
        print(f"  肿瘤药物数量: {len(tumor_drugs)}")
        
        # 获取细胞系药物列表
        cell_drugs = set()
        if self.cell_drug_dir.exists():
            pkl_files = [f for f in self.cell_drug_dir.iterdir() if f.name.endswith('_drug.pkl')]
            for pkl_file in pkl_files:
                drug_name = pkl_file.stem.replace('_drug', '')
                cell_drugs.add(drug_name)
        
        print(f"  细胞系药物数量: {len(cell_drugs)}")
        
        # 计算交集
        self.common_drugs = tumor_drugs.intersection(cell_drugs)
        print(f"  共同药物数量: {len(self.common_drugs)}")
        
        if self.common_drugs:
            print("  共同药物列表:")
            for drug in sorted(self.common_drugs):
                print(f"    {drug}")
        else:
            print("  警告: 没有找到共同的药物")
    
    def load_nadata(self, file_path):
        """加载nadata对象"""
        try:
            # 尝试使用nadata的load方法
            try:
                nadata_obj = nadata()
                nadata_obj.load(str(file_path))
                return nadata_obj
            except:
                # 如果失败，尝试直接pickle加载
                with open(file_path, 'rb') as f:
                    nadata_obj = pickle.load(f)
                return nadata_obj
        except Exception as e:
            print(f"  加载 {file_path} 时出错: {e}")
            return None
    
    def align_single_drug(self, drug_name):
        """
        对齐单个药物的数据，分别构建对齐后的nadata对象
        
        Args:
            drug_name: 药物名称
            
        Returns:
            tuple: (drug_name, success, (tumor_nadata, cell_nadata) or error_message)
        """
        try:
            print(f"开始对齐 {drug_name}...")
            
            # 构建文件路径
            tumor_file = self.tumor_drug_dir / f"{drug_name}_drug.pkl"
            cell_file = self.cell_drug_dir / f"{drug_name}_drug.pkl"
            
            # 检查文件是否存在
            if not tumor_file.exists():
                return drug_name, False, f"肿瘤药物文件不存在: {tumor_file}"
            
            if not cell_file.exists():
                return drug_name, False, f"细胞系药物文件不存在: {cell_file}"
            
            # 加载数据
            print(f"  加载肿瘤药物数据...")
            tumor_data = self.load_nadata(tumor_file)
            if tumor_data is None:
                return drug_name, False, f"无法加载肿瘤药物数据"
            
            print(f"  加载细胞系药物数据...")
            cell_data = self.load_nadata(cell_file)
            if cell_data is None:
                return drug_name, False, f"无法加载细胞系药物数据"
            
            print(f"  肿瘤数据形状: {tumor_data.X.shape}")
            print(f"  细胞系数据形状: {cell_data.X.shape}")
            
            # 获取基因列表
            tumor_genes = set(tumor_data.Var['Gene'].tolist())
            cell_genes = set(cell_data.Var['Gene'].tolist())
            
            print(f"  肿瘤基因数量: {len(tumor_genes)}")
            print(f"  细胞系基因数量: {len(cell_genes)}")
            
            # 计算基因交集
            common_genes = tumor_genes.intersection(cell_genes)
            print(f"  共同基因数量: {len(common_genes)}")
            
            if len(common_genes) < 10:
                return drug_name, False, f"共同基因数量不足: {len(common_genes)}"
            
            # 对齐基因数据
            common_genes_list = list(common_genes)
            common_genes_list.sort()  # 确保基因顺序一致
            
            # 过滤肿瘤数据
            tumor_var_filtered = tumor_data.Var[tumor_data.Var['Gene'].isin(common_genes_list)].copy()
            tumor_var_filtered = tumor_var_filtered.sort_values('Gene').reset_index(drop=True)
            
            # 过滤细胞系数据
            cell_var_filtered = cell_data.Var[cell_data.Var['Gene'].isin(common_genes_list)].copy()
            cell_var_filtered = cell_var_filtered.sort_values('Gene').reset_index(drop=True)
            
            # 确保基因顺序一致
            tumor_var_filtered = tumor_var_filtered.set_index('Gene').loc[common_genes_list].reset_index()
            cell_var_filtered = cell_var_filtered.set_index('Gene').loc[common_genes_list].reset_index()
            
            # 对齐肿瘤表达数据
            tumor_gene_indices = [i for i, gene in enumerate(tumor_data.Var['Gene']) if gene in common_genes_list]
            tumor_X = tumor_data.X[:, tumor_gene_indices]
            
            # 对齐细胞系表达数据
            cell_gene_indices = [i for i, gene in enumerate(cell_data.Var['Gene']) if gene in common_genes_list]
            cell_X = cell_data.X[:, cell_gene_indices]
            
            # 确保基因顺序一致
            tumor_gene_order = [tumor_data.Var.iloc[i]['Gene'] for i in tumor_gene_indices]
            cell_gene_order = [cell_data.Var.iloc[i]['Gene'] for i in cell_gene_indices]
            
            # 重新排列肿瘤数据
            tumor_gene_to_idx = {gene: idx for idx, gene in enumerate(tumor_gene_order)}
            tumor_reorder_indices = [tumor_gene_to_idx[gene] for gene in common_genes_list]
            tumor_X = tumor_X[:, tumor_reorder_indices]
            
            # 重新排列细胞系数据
            cell_gene_to_idx = {gene: idx for idx, gene in enumerate(cell_gene_order)}
            cell_reorder_indices = [cell_gene_to_idx[gene] for gene in common_genes_list]
            cell_X = cell_X[:, cell_reorder_indices]
            
            # 创建对齐后的肿瘤nadata对象
            aligned_tumor_nadata = nadata(
                X=tumor_X,
                Meta=tumor_data.Meta.copy(),
                Var=tumor_var_filtered,
                Prior=None
            )
            
            # 创建对齐后的细胞系nadata对象
            aligned_cell_nadata = nadata(
                X=cell_X,
                Meta=cell_data.Meta.copy(),
                Var=cell_var_filtered,
                Prior=None
            )
            
            # 设置肿瘤数据的训练/测试/验证索引
            n_tumor_samples = tumor_X.shape[0]
            tumor_indices = np.arange(n_tumor_samples)
            np.random.seed(42)  # 设置随机种子
            np.random.shuffle(tumor_indices)
            
            # 70% 训练, 15% 验证, 15% 测试
            train_size = int(0.7 * n_tumor_samples)
            val_size = int(0.15 * n_tumor_samples)
            
            train_idx = tumor_indices[:train_size]
            val_idx = tumor_indices[train_size:train_size + val_size]
            test_idx = tumor_indices[train_size + val_size:]
            
            aligned_tumor_nadata.Model.set_indices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
            
            # 设置细胞系数据的训练/测试/验证索引
            n_cell_samples = cell_X.shape[0]
            cell_indices = np.arange(n_cell_samples)
            np.random.seed(42)  # 设置随机种子
            np.random.shuffle(cell_indices)
            
            # 70% 训练, 15% 验证, 15% 测试
            train_size = int(0.7 * n_cell_samples)
            val_size = int(0.15 * n_cell_samples)
            
            train_idx = cell_indices[:train_size]
            val_idx = cell_indices[train_size:train_size + val_size]
            test_idx = cell_indices[train_size + val_size:]
            
            aligned_cell_nadata.Model.set_indices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
            
            print(f"  {drug_name} 对齐完成:")
            print(f"    肿瘤样本: {tumor_X.shape[0]}, 基因: {tumor_X.shape[1]}")
            print(f"    细胞系样本: {cell_X.shape[0]}, 基因: {cell_X.shape[1]}")
            
            return drug_name, True, (aligned_tumor_nadata, aligned_cell_nadata)
            
        except Exception as e:
            error_msg = f"对齐 {drug_name} 时出错: {str(e)}"
            print(f"  {error_msg}")
            return drug_name, False, error_msg
    
    def align_all_drugs(self):
        """对齐所有共同药物"""
        print(f"开始对齐 {len(self.common_drugs)} 种共同药物...")
        
        start_time = time.time()
        
        completed = 0
        successful = 0
        failed_reasons = defaultdict(int)  # 统计失败原因
        
        # 逐个处理每个药物
        for drug_name in sorted(self.common_drugs):
            try:
                drug_name, success, result = self.align_single_drug(drug_name)
                completed += 1
                
                if success:
                    successful += 1
                    # 保存数据
                    self._save_aligned_data(drug_name, result)
                else:
                    # 统计失败原因
                    failed_reasons[result] += 1
                
                print(f"进度: {completed}/{len(self.common_drugs)} ({successful} 成功)")
                
            except Exception as e:
                completed += 1
                error_msg = f"处理异常: {str(e)}"
                failed_reasons[error_msg] += 1
                print(f"对齐 {drug_name} 时发生异常: {e}")
        
        end_time = time.time()
        print(f"\n对齐完成!")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"成功对齐: {successful}/{len(self.common_drugs)} 种药物")
        
        # 打印失败原因统计
        if failed_reasons:
            print(f"\n失败原因统计:")
            for reason, count in failed_reasons.items():
                print(f"  {reason}: {count} 个药物")
    
    def _save_aligned_data(self, drug_name, nadata_objects):
        """保存对齐后的数据到原来的文件中"""
        try:
            aligned_tumor_nadata, aligned_cell_nadata = nadata_objects
            
            # 创建安全的文件名
            safe_name = drug_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
            
            # 保存肿瘤数据到原来的文件
            tumor_output_path = self.tumor_drug_dir / f"{safe_name}_drug.pkl"
            try:
                aligned_tumor_nadata.save(str(tumor_output_path), format='pickle', save_data=True)
                print(f"  肿瘤数据已保存到: {tumor_output_path}")
            except Exception as e:
                print(f"  保存肿瘤数据时出错: {e}")
                # 尝试使用pickle直接保存
                try:
                    with open(tumor_output_path, 'wb') as f:
                        pickle.dump(aligned_tumor_nadata, f)
                    print(f"  使用pickle直接保存肿瘤数据成功: {tumor_output_path}")
                except Exception as e2:
                    print(f"  pickle保存肿瘤数据也失败: {e2}")
                    return
            
            # 保存细胞系数据到原来的文件
            cell_output_path = self.cell_drug_dir / f"{safe_name}_drug.pkl"
            try:
                aligned_cell_nadata.save(str(cell_output_path), format='pickle', save_data=True)
                print(f"  细胞系数据已保存到: {cell_output_path}")
            except Exception as e:
                print(f"  保存细胞系数据时出错: {e}")
                # 尝试使用pickle直接保存
                try:
                    with open(cell_output_path, 'wb') as f:
                        pickle.dump(aligned_cell_nadata, f)
                    print(f"  使用pickle直接保存细胞系数据成功: {cell_output_path}")
                except Exception as e2:
                    print(f"  pickle保存细胞系数据也失败: {e2}")
                    return
            
            # 记录成功信息
            self.results[safe_name] = {
                'drug_name': drug_name,
                'tumor_file_path': str(tumor_output_path),
                'cell_file_path': str(cell_output_path),
                'tumor_shape': aligned_tumor_nadata.X.shape,
                'cell_shape': aligned_cell_nadata.X.shape,
                'tumor_meta_shape': aligned_tumor_nadata.Meta.shape,
                'cell_meta_shape': aligned_cell_nadata.Meta.shape,
                'tumor_var_shape': aligned_tumor_nadata.Var.shape,
                'cell_var_shape': aligned_cell_nadata.Var.shape
            }
                    
        except Exception as e:
            print(f"  保存 {drug_name} 数据时出错: {e}")
    
    def get_alignment_summary(self):
        """获取对齐摘要"""
        print("\n对齐摘要:")
        print(f"共同药物数量: {len(self.common_drugs)}")
        
        for drug in sorted(self.common_drugs):
            print(f"  {drug}")
        
        print(f"\n成功对齐: {len(self.results)}")
        for safe_name, info in self.results.items():
            print(f"  {info['drug_name']}:")
            print(f"    肿瘤数据: {info['tumor_shape']}")
            print(f"    细胞系数据: {info['cell_shape']}")
        
        # 计算失败数量
        failed_count = len(self.common_drugs) - len(self.results)
        if failed_count > 0:
            print(f"\n失败对齐: {failed_count}")
            print("失败原因可能包括:")
            print("  - 文件不存在")
            print("  - 无法加载数据")
            print("  - 共同基因数量不足")
            print("  - 数据格式不一致")

def load_aligned_data(drug_name=None):
    """
    加载对齐后的数据
    
    Args:
        drug_name: 药物名称，如果为None则返回所有可用的药物数据
        
    Returns:
        dict: 药物数据字典
    """
    # 使用绝对路径
    base_dir = Path(__file__).parent
    tumor_drug_dir = base_dir / "datasets" / "tumor_drug"
    cell_drug_dir = base_dir / "datasets" / "cell_drug"
    
    if not tumor_drug_dir.exists():
        raise FileNotFoundError(f"肿瘤药物数据目录不存在: {tumor_drug_dir}")
    
    if not cell_drug_dir.exists():
        raise FileNotFoundError(f"细胞系药物数据目录不存在: {cell_drug_dir}")
    
    # 查找所有pkl文件
    tumor_pkl_files = [f for f in tumor_drug_dir.iterdir() if f.name.endswith('_drug.pkl')]
    cell_pkl_files = [f for f in cell_drug_dir.iterdir() if f.name.endswith('_drug.pkl')]
    
    if not tumor_pkl_files:
        raise FileNotFoundError(f"在 {tumor_drug_dir} 中未找到任何肿瘤药物数据文件")
    
    if not cell_pkl_files:
        raise FileNotFoundError(f"在 {cell_drug_dir} 中未找到任何细胞系药物数据文件")
    
    results = {}
    
    # 获取所有药物名称
    tumor_drugs = {f.stem.replace('_drug', '') for f in tumor_pkl_files}
    cell_drugs = {f.stem.replace('_drug', '') for f in cell_pkl_files}
    common_drugs = tumor_drugs.intersection(cell_drugs)
    
    for drug_name_file in common_drugs:
        if drug_name is None or drug_name.lower() in drug_name_file.lower():
            try:
                print(f"加载 {drug_name_file} 对齐数据...")
                
                # 加载肿瘤数据
                tumor_file = tumor_drug_dir / f"{drug_name_file}_drug.pkl"
                try:
                    tumor_nadata_obj = nadata()
                    tumor_nadata_obj.load(str(tumor_file))
                except:
                    with open(tumor_file, 'rb') as f:
                        tumor_nadata_obj = pickle.load(f)
                
                # 加载细胞系数据
                cell_file = cell_drug_dir / f"{drug_name_file}_drug.pkl"
                try:
                    cell_nadata_obj = nadata()
                    cell_nadata_obj.load(str(cell_file))
                except:
                    with open(cell_file, 'rb') as f:
                        cell_nadata_obj = pickle.load(f)
                
                results[drug_name_file] = {
                    'tumor_data': tumor_nadata_obj,
                    'cell_data': cell_nadata_obj,
                    'tumor_file_path': str(tumor_file),
                    'cell_file_path': str(cell_file),
                    'tumor_shape': tumor_nadata_obj.X.shape,
                    'cell_shape': cell_nadata_obj.X.shape,
                    'tumor_meta_shape': tumor_nadata_obj.Meta.shape,
                    'cell_meta_shape': cell_nadata_obj.Meta.shape,
                    'tumor_var_shape': tumor_nadata_obj.Var.shape,
                    'cell_var_shape': cell_nadata_obj.Var.shape
                }
                
                print(f"  {drug_name_file} 加载成功:")
                print(f"    肿瘤数据: {tumor_nadata_obj.X.shape}")
                print(f"    细胞系数据: {cell_nadata_obj.X.shape}")
                
            except Exception as e:
                print(f"  加载 {drug_name_file} 时出错: {e}")
    
    return results

if __name__ == "__main__":
    try:
        print("=" * 50)
        print("肿瘤和细胞系药物数据对齐程序开始运行")
        print("=" * 50)
        
        # 创建对齐器并处理所有药物
        aligner = TumorCellDrugAligner()
        
        # 显示对齐摘要
        aligner.get_alignment_summary()
        
        # 开始对齐
        aligner.align_all_drugs()
        
        # 测试加载
        print("\n测试对齐数据加载...")
        try:
            loaded_data = load_aligned_data()
            print(f"成功加载 {len(loaded_data)} 个对齐药物数据集")
            
            # 显示详细信息
            for drug_name, info in loaded_data.items():
                print(f"\n{drug_name}:")
                print(f"  肿瘤数据形状: {info['tumor_shape']}")
                print(f"  细胞系数据形状: {info['cell_shape']}")
                print(f"  肿瘤表型数据: {info['tumor_meta_shape']}")
                print(f"  细胞系表型数据: {info['cell_meta_shape']}")
                print(f"  肿瘤基因数据: {info['tumor_var_shape']}")
                print(f"  细胞系基因数据: {info['cell_var_shape']}")
                
        except Exception as e:
            print(f"加载数据时出错: {e}")
        
        print("\n数据对齐完成!")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
