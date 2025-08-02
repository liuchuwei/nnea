#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCGA肿瘤生存数据处理脚本
多线程批量处理样本数大于100的肿瘤生存数据
构建nnea nadata对象，保存到datasets/tumor_survival文件夹中
"""

import warnings
import pandas as pd
import numpy as np
import os
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pickle
import subprocess

# 忽略pandas的pyarrow警告
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='pyarrow')

# 添加nnea模块路径
sys.path.append(str(Path(__file__).parent))

from nnea.io._nadata import nadata

class TumorSurvivalProcessor:
    """肿瘤生存数据处理器"""
    
    def __init__(self, max_workers=4):
        """
        初始化处理器
        
        Args:
            max_workers: 最大线程数
        """
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.results = {}
        
        # 数据文件路径
        self.tumor_exp_path = "../data/tumor/tcga/Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
        self.hugo_geneset_path = "../data/hugo_2025.txt"
        self.sur_dat_path = "../data/tumor/tcga/TCGA_survival_data_2.txt"
        self.phenotype_path = "../data/tumor/tcga/TCGA_TARGET_phenotype.txt"
        
        # 检查文件是否存在
        self._check_files()
        
        # 加载共享数据
        self._load_shared_data()
    
    def _check_files(self):
        """检查必要文件是否存在"""
        required_files = [
            self.tumor_exp_path,
            self.hugo_geneset_path,
            self.sur_dat_path,
            self.phenotype_path
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(f"以下文件不存在:\n" + "\n".join(missing_files))
    
    def _load_shared_data(self):
        """加载共享数据"""
        print("加载共享数据...")
        
        # 加载生存数据
        print("  加载生存数据...")
        self.sur_dat = pd.read_csv(self.sur_dat_path, sep='\t')
        print(f"  生存数据形状: {self.sur_dat.shape}")
        
        # 加载表型数据
        print("  加载表型数据...")
        self.phenotype = pd.read_csv(self.phenotype_path, sep='\t')
        print(f"  表型数据形状: {self.phenotype.shape}")
        
        # 加载HUGO基因集
        print("  加载HUGO基因集...")
        self.hugo_geneset = pd.read_csv(self.hugo_geneset_path, sep='\t')
        print(f"  HUGO基因集形状: {self.hugo_geneset.shape}")
        
        # 获取所有可用的肿瘤类型
        print("  分析肿瘤类型...")
        self.tumor_cohorts = self._get_tumor_cohorts()
        print(f"  发现 {len(self.tumor_cohorts)} 种肿瘤类型")
    
    def _get_tumor_cohorts(self):
        """获取所有肿瘤类型及其样本数"""
        cohort_counts = self.phenotype['_cohort'].value_counts()
        
        # 过滤样本数大于100的肿瘤类型
        valid_cohorts = {}
        for cohort, count in cohort_counts.items():
            if count >= 100:  # 只处理样本数大于100的肿瘤
                # 检查是否有生存数据
                cohort_samples = set(self.phenotype[self.phenotype['_cohort'] == cohort]['sampleID'])
                sur_samples = set(self.sur_dat['sample'])
                common_samples = cohort_samples.intersection(sur_samples)
                
                if len(common_samples) >= 50:  # 至少50个样本有生存数据
                    valid_cohorts[cohort] = {
                        'total_samples': count,
                        'survival_samples': len(common_samples)
                    }
        
        return valid_cohorts
    
    def process_single_tumor(self, cohort_name):
        """
        处理单个肿瘤类型的数据
        
        Args:
            cohort_name: 肿瘤类型名称
            
        Returns:
            tuple: (cohort_name, success, nadata_obj or error_message)
        """
        try:
            print(f"开始处理 {cohort_name}...")
            
            # 获取该肿瘤类型的样本
            tumor_samples = self.phenotype[self.phenotype['_cohort'] == cohort_name].copy()
            tumor_sample_ids = set(tumor_samples['sampleID'])
            sur_sample_ids = set(self.sur_dat['sample'])
            common_sample_ids = tumor_sample_ids.intersection(sur_sample_ids)
            
            if len(common_sample_ids) < 50:
                return cohort_name, False, f"样本数不足: {len(common_sample_ids)}"
            
            # 构建生存数据
            sur = self.sur_dat[self.sur_dat['sample'].isin(common_sample_ids)].copy()
            sur = sur[['sample', 'OS', 'OS.time']].copy()
            sur = sur.dropna()
            
            if len(sur) < 30:
                return cohort_name, False, f"有效生存数据不足: {len(sur)}"
            
            # 构建表型数据
            meta1 = self.phenotype[self.phenotype['sampleID'].isin(sur['sample'])].copy()
            meta2 = self.sur_dat[self.sur_dat['sample'].isin(sur['sample'])].copy()
            
            # 合并表型数据
            meta1 = meta1.reset_index(drop=True)
            meta2 = meta2.reset_index(drop=True)
            meta1['id'] = meta1['sampleID']
            meta2_subset = meta2.drop('sample', axis=1)
            
            if len(meta1) == len(meta2_subset):
                Meta = pd.concat([meta1, meta2_subset], axis=1)
            else:
                Meta = sur.copy()
                Meta['id'] = Meta['sample']
                for col in ['_cohort', '_primary_site', '_primary_disease', 'age_at_initial_pathologic_diagnosis', 'gender']:
                    if col in self.phenotype.columns:
                        Meta[col] = Meta['sample'].map(self.phenotype.set_index('sampleID')[col])
            
            # 构建基因数据
            hugo_gene = self.hugo_geneset[self.hugo_geneset['locus_group'] == 'protein-coding gene'].copy()
            Var = hugo_gene[['symbol', 'name', 'ensembl_gene_id', 'location', 'locus_group']].copy()
            Var.columns = ['Gene', 'Name', 'ENS', 'location', 'locus_group']
            
            # 读取表达矩阵（只读取需要的样本和基因）
            try:
                # 读取表达矩阵，只选择需要的样本和基因
                exp = pd.read_csv(self.tumor_exp_path, sep='\t', usecols=['Gene'] + list(sur['sample']))
                
                # 过滤基因
                exp = exp[exp['Gene'].isin(Var['Gene'])].copy()
                
                # 准备表达矩阵X
                X = exp.set_index('Gene').T  # 转置，使行为样本，列为基因
                
            except Exception as e:
                print(f"  警告: 无法读取完整表达矩阵: {e}")
                # 创建空的表达矩阵
                X = pd.DataFrame(index=sur['sample'], columns=Var['Gene'])
                X = X.fillna(0)  # 填充0
            
            # 确保数据一致性
            common_samples = set(sur['sample']).intersection(set(X.index))
            common_genes = set(Var['Gene']).intersection(set(X.columns))
            
            sur = sur[sur['sample'].isin(common_samples)].copy()
            Meta = Meta[Meta['id'].isin(common_samples)].copy()
            X = X.loc[list(common_samples), list(common_genes)]
            Var = Var[Var['Gene'].isin(common_genes)].copy()
            
            if len(common_samples) < 20 or len(common_genes) < 100:
                return cohort_name, False, f"数据不足: 样本{len(common_samples)}, 基因{len(common_genes)}"
            
            # 创建nadata对象
            nadata_obj = nadata(
                X=X.values,  # 转换为numpy数组
                Meta=Meta,
                Var=Var,
                Prior=None
            )
            
            # 设置训练/测试/验证索引
            n_samples = len(sur)
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
            
            print(f"  {cohort_name} 处理完成: 样本{len(common_samples)}, 基因{len(common_genes)}")
            return cohort_name, True, nadata_obj
            
        except Exception as e:
            error_msg = f"处理 {cohort_name} 时出错: {str(e)}"
            print(f"  {error_msg}")
            return cohort_name, False, error_msg
    
    def process_all_tumors(self):
        """多线程处理所有肿瘤类型"""
        print(f"开始多线程处理 {len(self.tumor_cohorts)} 种肿瘤类型...")
        print(f"使用 {self.max_workers} 个线程")
        
        start_time = time.time()
        
        # 使用线程池处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_cohort = {
                executor.submit(self.process_single_tumor, cohort): cohort 
                for cohort in self.tumor_cohorts.keys()
            }
            
            # 收集结果
            completed = 0
            successful = 0
            
            for future in as_completed(future_to_cohort):
                cohort = future_to_cohort[future]
                try:
                    cohort_name, success, result = future.result()
                    completed += 1
                    
                    if success:
                        successful += 1
                        # 保存数据
                        self._save_tumor_data(cohort_name, result)
                    
                    print(f"进度: {completed}/{len(self.tumor_cohorts)} ({successful} 成功)")
                    
                except Exception as e:
                    completed += 1
                    print(f"处理 {cohort} 时发生异常: {e}")
        
        end_time = time.time()
        print(f"\n处理完成!")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"成功处理: {successful}/{len(self.tumor_cohorts)} 种肿瘤类型")
    
    def _save_tumor_data(self, cohort_name, nadata_obj):
        """保存肿瘤数据"""
        # 创建安全的文件名
        safe_name = cohort_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
        
        output_dir = "../datasets/tumor_survival"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为pickle格式
        output_path = os.path.join(output_dir, f"{safe_name}_survival.pkl")
        
        try:
            nadata_obj.save(output_path, format='pickle', save_data=True)
            print(f"  数据已保存到: {output_path}")
            
            # 记录成功信息
            with self.lock:
                self.results[safe_name] = {
                    'cohort_name': cohort_name,
                    'file_path': output_path,
                    'shape': nadata_obj.X.shape,
                    'meta_shape': nadata_obj.Meta.shape,
                    'var_shape': nadata_obj.Var.shape
                }
                
        except Exception as e:
            print(f"  保存 {cohort_name} 数据时出错: {e}")
    
    def get_processing_summary(self):
        """获取处理摘要"""
        print("\n处理摘要:")
        print(f"发现肿瘤类型: {len(self.tumor_cohorts)}")
        
        for cohort, info in self.tumor_cohorts.items():
            print(f"  {cohort}: 总样本{info['total_samples']}, 生存样本{info['survival_samples']}")
        
        print(f"\n成功处理: {len(self.results)}")
        for safe_name, info in self.results.items():
            print(f"  {info['cohort_name']}: {info['shape']}")

def load_tumor_survival_data(cohort_name=None):
    """
    加载肿瘤生存数据
    
    Args:
        cohort_name: 肿瘤类型名称，如果为None则返回所有可用的肿瘤类型
        
    Returns:
        dict: 肿瘤数据字典
    """
    output_dir = "../datasets/tumor_survival"
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"数据目录不存在: {output_dir}")
    
    # 查找所有pkl文件
    pkl_files = [f for f in os.listdir(output_dir) if f.endswith('_survival.pkl')]
    
    if not pkl_files:
        raise FileNotFoundError(f"在 {output_dir} 中未找到任何肿瘤生存数据文件")
    
    results = {}
    
    for pkl_file in pkl_files:
        file_path = os.path.join(output_dir, pkl_file)
        tumor_name = pkl_file.replace('_survival.pkl', '')
        
        if cohort_name is None or cohort_name.lower() in tumor_name.lower():
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
    # 创建处理器并处理所有肿瘤
    processor = TumorSurvivalProcessor(max_workers=4)
    
    # 显示处理摘要
    processor.get_processing_summary()
    
    # 开始处理
    processor.process_all_tumors()
    
    # 测试加载
    print("\n测试数据加载...")
    try:
        loaded_data = load_tumor_survival_data()
        print(f"成功加载 {len(loaded_data)} 个肿瘤数据集")
    except Exception as e:
        print(f"加载数据时出错: {e}")
    
    print("\n数据处理完成!")
