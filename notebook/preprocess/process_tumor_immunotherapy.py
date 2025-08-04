#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肿瘤免疫治疗数据处理脚本
处理melanoma, ccRCC, bladder不同对象的 *.phe.txt及*.exp.txt文件
构建nnea nadata对象，保存到datasets/tumor_imm文件夹中
"""

import warnings
import pandas as pd
import numpy as np
import os
import sys
import threading
import time
from pathlib import Path
from collections import defaultdict
import pickle
import subprocess

# 忽略pandas的pyarrow警告
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='pyarrow')

# 添加nnea模块路径
sys.path.append(str(Path(__file__).parent))

print("开始导入nnea模块...")
try:
    from nnea.io import nadata
    print("nnea模块导入成功")
except Exception as e:
    print(f"nnea模块导入失败: {e}")
    sys.exit(1)

print("脚本开始运行...")

class TumorImmunotherapyProcessor:
    """肿瘤免疫治疗数据处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.lock = threading.Lock()
        self.results = {}
        
        # 数据文件路径
        self.data_path = "../../data/tumor/immune_therapy"
        self.output_path = "../../datasets/tumor_imm"
        
        # 检查文件是否存在
        self._check_files()
        
        # 加载HUGO基因集
        self._load_hugo_geneset()
    
    def _check_files(self):
        """检查必要文件是否存在"""
        required_files = [
            f"{self.data_path}/imm_melanoma_exp.txt",
            f"{self.data_path}/imm_melanoma_phe.txt",
            f"{self.data_path}/imm_ccRCC_exp.txt",
            f"{self.data_path}/imm_ccRCC_phe.txt",
            f"{self.data_path}/imm_bladder_exp.txt",
            f"{self.data_path}/imm_bladder_phe.txt",
            "data/hugo_2025.txt"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"警告: 以下文件不存在:")
            for file_path in missing_files:
                print(f"  {file_path}")
            print("将尝试处理可用的文件...")
    
    def _load_hugo_geneset(self):
        """加载HUGO基因集"""
        print("加载HUGO基因集...")
        try:
            self.hugo_geneset = pd.read_csv("../../data/hugo_2025.txt", sep='\t')
            print(f"  HUGO基因集形状: {self.hugo_geneset.shape}")
        except Exception as e:
            print(f"  警告: 无法加载HUGO基因集: {e}")
            self.hugo_geneset = None
    
    def process_melanoma(self):
        """处理melanoma数据"""
        print("处理melanoma数据...")
        
        try:
            # 读取表达数据
            exp_file = f"{self.data_path}/imm_melanoma_exp.txt"
            if not os.path.exists(exp_file):
                print(f"  文件不存在: {exp_file}")
                return False, "文件不存在"
            
            # 使用逗号分隔符读取数据
            exp = pd.read_csv(exp_file, sep=',')
            print(f"  表达数据形状: {exp.shape}")
            
            # 读取表型数据
            phe_file = f"{self.data_path}/imm_melanoma_phe.txt"
            if not os.path.exists(phe_file):
                print(f"  文件不存在: {phe_file}")
                return False, "文件不存在"
            
            # 使用逗号分隔符读取数据
            phe = pd.read_csv(phe_file, sep=',')
            print(f"  表型数据形状: {phe.shape}")
            
            # 构建nadata对象
            nadata_obj = self._build_nadata_object(exp, phe, "melanoma")
            
            # 保存数据
            self._save_tumor_data("melanoma", nadata_obj)
            
            return True, nadata_obj
            
        except Exception as e:
            error_msg = f"处理melanoma数据时出错: {str(e)}"
            print(f"  {error_msg}")
            return False, error_msg
    
    def process_ccRCC(self):
        """处理ccRCC数据"""
        print("处理ccRCC数据...")
        
        try:
            # 读取表达数据
            exp_file = f"{self.data_path}/imm_ccRCC_exp.txt"
            if not os.path.exists(exp_file):
                print(f"  文件不存在: {exp_file}")
                return False, "文件不存在"
            
            # 使用逗号分隔符读取数据
            exp = pd.read_csv(exp_file, sep=',')
            print(f"  表达数据形状: {exp.shape}")
            
            # 读取表型数据
            phe_file = f"{self.data_path}/imm_ccRCC_phe.txt"
            if not os.path.exists(phe_file):
                print(f"  文件不存在: {phe_file}")
                return False, "文件不存在"
            
            # 使用逗号分隔符读取数据
            phe = pd.read_csv(phe_file, sep=',')
            print(f"  表型数据形状: {phe.shape}")
            
            # 构建nadata对象
            nadata_obj = self._build_nadata_object(exp, phe, "ccRCC")
            
            # 保存数据
            self._save_tumor_data("ccRCC", nadata_obj)
            
            return True, nadata_obj
            
        except Exception as e:
            error_msg = f"处理ccRCC数据时出错: {str(e)}"
            print(f"  {error_msg}")
            return False, error_msg
    
    def process_bladder(self):
        """处理bladder数据"""
        print("处理bladder数据...")
        
        try:
            # 读取bladder表达数据
            exp_file = f"{self.data_path}/imm_bladder_exp.txt"
            phe_file = f"{self.data_path}/imm_bladder_phe.txt"
            
            # 检查文件是否存在
            if not os.path.exists(exp_file) or not os.path.exists(phe_file):
                print(f"  bladder文件不存在: {exp_file} 或 {phe_file}")
                return False, "文件不存在"
            
            # 读取数据
            exp = pd.read_csv(exp_file, sep=',')
            phe = pd.read_csv(phe_file, sep=',')
            
            print(f"  bladder表达数据形状: {exp.shape}")
            print(f"  bladder表型数据形状: {phe.shape}")
            
            # 构建nadata对象
            nadata_obj = self._build_nadata_object(exp, phe, "bladder")
            
            # 保存数据
            self._save_tumor_data("bladder", nadata_obj)
            
            return True, nadata_obj
            
        except Exception as e:
            error_msg = f"处理bladder数据时出错: {str(e)}"
            print(f"  {error_msg}")
            return False, error_msg
    

    
    def _build_nadata_object(self, exp, phe, dataset_name):
        """构建nadata对象"""
        print(f"  构建 {dataset_name} 的nadata对象...")
        
        # 准备表达矩阵X
        if exp.shape[1] > 1:  # 确保有基因列和样本列
            gene_col = exp.columns[0]
            sample_cols = exp.columns[1:]
            
            # 设置基因为索引
            exp_matrix = exp.set_index(gene_col)
            X = exp_matrix.T  # 转置，使行为样本，列为基因
            
            # 准备基因信息Var
            if self.hugo_geneset is not None:
                hugo_gene = self.hugo_geneset[self.hugo_geneset['locus_group'] == 'protein-coding gene'].copy()
                Var = hugo_gene[['symbol', 'name', 'ensembl_gene_id', 'location', 'locus_group']].copy()
                Var.columns = ['Gene', 'Name', 'ENS', 'location', 'locus_group']
                
                # 过滤基因
                common_genes = set(X.columns).intersection(set(Var['Gene']))
                if len(common_genes) > 0:
                    X = X[list(common_genes)]
                    Var = Var[Var['Gene'].isin(common_genes)].copy()
                else:
                    print("    警告: 没有找到匹配的HUGO基因，使用所有基因")
                    Var = pd.DataFrame({
                        'Gene': X.columns,
                        'Name': X.columns,
                        'ENS': [''] * len(X.columns),
                        'location': [''] * len(X.columns),
                        'locus_group': ['protein-coding gene'] * len(X.columns)
                    })
            else:
                # 如果没有HUGO基因集，创建简单的基因信息
                Var = pd.DataFrame({
                    'Gene': X.columns,
                    'Name': X.columns,
                    'ENS': [''] * len(X.columns),
                    'location': [''] * len(X.columns),
                    'locus_group': ['protein-coding gene'] * len(X.columns)
                })
        else:
            print("  警告: 表达数据格式不正确")
            return None
        
        # 准备表型数据Meta
        Meta = phe.copy()
        
        # 确保样本ID列存在
        if 'sample_id' in Meta.columns:
            Meta['id'] = Meta['sample_id']
        elif 'id' not in Meta.columns:
            # 如果没有样本ID列，使用行索引
            Meta['id'] = [f"sample_{i}" for i in range(len(Meta))]
        
        # 尝试不同的样本ID列名
        possible_id_cols = ['id', 'sample_id', 'sample', 'V1', 'ID']
        sample_id_col = None
        for col in possible_id_cols:
            if col in Meta.columns:
                sample_id_col = col
                break
        
        if sample_id_col is None:
            # 如果没有找到样本ID列，使用第一列
            sample_id_col = Meta.columns[0]
            Meta['id'] = Meta[sample_id_col]
        
        # 确保数据一致性
        common_samples = set(X.index).intersection(set(Meta['id']))
        
        if len(common_samples) < 10:
            print(f"  警告: 共同样本数过少: {len(common_samples)}")
            print(f"    X样本数: {len(X.index)}")
            print(f"    Meta样本数: {len(Meta)}")
            print(f"    X样本示例: {list(X.index)[:5]}")
            print(f"    Meta样本示例: {list(Meta['id'])[:5]}")
            
            # 尝试使用所有样本，不进行匹配
            if len(X.index) >= 10:
                print("    使用所有表达数据样本")
                X = X.copy()
                Meta = Meta.copy()
                # 为Meta添加缺失的样本
                missing_samples = set(X.index) - set(Meta['id'])
                for sample in missing_samples:
                    new_row = pd.Series({'id': sample})
                    Meta = pd.concat([Meta, pd.DataFrame([new_row])], ignore_index=True)
            else:
                return None
        
        X = X.loc[list(common_samples)]
        Meta = Meta[Meta['id'].isin(common_samples)].copy()
        
        print(f"    最终数据形状: X={X.shape}, Meta={Meta.shape}, Var={Var.shape}")
        
        # 创建nadata对象
        nadata_obj = nadata(
            X=X.values,  # 转换为numpy数组
            Meta=Meta,
            Var=Var,
            Prior=None
        )
        
        # 设置训练/测试/验证索引
        n_samples = len(common_samples)
        indices = np.arange(n_samples)
        np.random.seed(42)  # 设置随机种子
        np.random.shuffle(indices)
        
        # 70% 训练, 15% 验证, 15% 测试
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        nadata_obj.Model.set_indices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        
        return nadata_obj
    
    def _save_tumor_data(self, dataset_name, nadata_obj):
        """保存肿瘤数据"""
        # 创建输出目录
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存为pickle格式
        output_path = os.path.join(self.output_path, f"{dataset_name}_immunotherapy.pkl")
        
        try:
            nadata_obj.save(output_path, format='pickle', save_data=True)
            print(f"  数据已保存到: {output_path}")
            
            # 记录成功信息
            with self.lock:
                self.results[dataset_name] = {
                    'file_path': output_path,
                    'shape': nadata_obj.X.shape,
                    'meta_shape': nadata_obj.Meta.shape,
                    'var_shape': nadata_obj.Var.shape
                }
                
        except Exception as e:
            print(f"  保存 {dataset_name} 数据时出错: {e}")
    
    def process_all_datasets(self):
        """处理所有数据集"""
        print("开始处理肿瘤免疫治疗数据...")
        
        start_time = time.time()
        
        # 处理melanoma
        success1, result1 = self.process_melanoma()
        
        # 处理ccRCC
        success2, result2 = self.process_ccRCC()
        
        # 处理bladder
        success3, result3 = self.process_bladder()
        
        end_time = time.time()
        
        print(f"\n处理完成!")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"成功处理: {sum([success1, success2, success3])}/3 个数据集")
        
        # 显示处理摘要
        self.get_processing_summary()
    
    def get_processing_summary(self):
        """获取处理摘要"""
        print("\n处理摘要:")
        print(f"成功处理数据集: {len(self.results)}")
        
        for dataset_name, info in self.results.items():
            print(f"  {dataset_name}: {info['shape']}")

def load_tumor_immunotherapy_data(dataset_name=None):
    """
    加载肿瘤免疫治疗数据
    
    Args:
        dataset_name: 数据集名称，如果为None则返回所有可用的数据集
        
    Returns:
        dict: 肿瘤数据字典
    """
    output_dir = "../../datasets/tumor_imm"
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"数据目录不存在: {output_dir}")
    
    # 查找所有pkl文件
    pkl_files = [f for f in os.listdir(output_dir) if f.endswith('_immunotherapy.pkl')]
    
    if not pkl_files:
        raise FileNotFoundError(f"在 {output_dir} 中未找到任何肿瘤免疫治疗数据文件")
    
    results = {}
    
    for pkl_file in pkl_files:
        file_path = os.path.join(output_dir, pkl_file)
        tumor_name = pkl_file.replace('_immunotherapy.pkl', '')
        
        if dataset_name is None or dataset_name.lower() in tumor_name.lower():
            try:
                print(f"加载 {tumor_name} 数据...")
                nadata_obj = nadata()
                nadata_obj.load(file_path)
                
                results[tumor_name] = {
                    'data': nadata_obj,
                    'file_path': file_path,
                    'shape': nadata_obj.X.shape,
                    'meta_shape': nadata_obj.Meta.shape,
                    'var_shape': nadata_obj.Var.shape
                }
                
                print(f"  {tumor_name} 加载成功: {nadata_obj.X.shape}")
                
            except Exception as e:
                print(f"  加载 {tumor_name} 时出错: {e}")
    
    return results

if __name__ == "__main__":
    # 创建处理器并处理所有数据集
    processor = TumorImmunotherapyProcessor()
    
    # 开始处理
    processor.process_all_datasets()
    
    # 测试加载
    print("\n测试数据加载...")
    try:
        loaded_data = load_tumor_immunotherapy_data()
        print(f"成功加载 {len(loaded_data)} 个肿瘤免疫治疗数据集")
    except Exception as e:
        print(f"加载数据时出错: {e}")
    
    print("\n数据处理完成!")
