#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
细胞系-肿瘤依赖关系数据处理脚本
将R脚本转换为Python版本，处理细胞系和肿瘤的表达数据
构建nadata对象，X为cell_line的表达矩阵，Meta为celldep
tumor_dat和cell_info保存为csv格式
"""

import warnings
import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
import pickle
import re

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

class CellTumorDependencyProcessor:
    """细胞系-肿瘤依赖关系数据处理器"""
    
    def __init__(self):
        """
        初始化处理器
        """
        self.results = {}
        
        # 数据文件路径
        base_dir = Path(__file__).parent
        self.tumor_data_path = base_dir / "data" / "tumor" / "tcga" / "Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
        self.cell_data_path = base_dir / "data" / "cell_line" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
        self.hugo_path = base_dir / "data" / "hugo_2025.txt"
        self.celldep_path = base_dir / "data" / "cell_line" / "CRISPRGeneDependency.csv"
        self.cell_info_path = base_dir / "data" / "cell_line" / "Celligner_info.csv"
        
        # 输出路径
        self.output_dir = base_dir / "datasets" / "tumor_cell_dep"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查文件是否存在
        self._check_files()
        
        # 加载共享数据
        self._load_shared_data()
    
    def _check_files(self):
        """检查必要文件是否存在"""
        required_files = [
            self.tumor_data_path,
            self.cell_data_path,
            self.hugo_path,
            self.celldep_path,
            self.cell_info_path
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            print("警告: 以下文件不存在:")
            for file_path in missing_files:
                print(f"  {file_path}")
            print("请确保数据文件已正确放置")
            # 不抛出异常，而是使用默认值
            self._create_default_data()
        else:
            print("所有必要文件检查通过")
    
    def _create_default_data(self):
        """创建默认数据用于测试"""
        print("创建默认测试数据...")
        
        # 创建默认的肿瘤数据
        genes = ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5']
        samples = ['TCGA-01', 'TCGA-02', 'TCGA-03']
        
        np.random.seed(42)
        tumor_data = pd.DataFrame(
            np.random.randn(len(genes), len(samples)),
            index=genes,
            columns=samples
        )
        tumor_data = tumor_data.reset_index()
        tumor_data.columns = ['Gene'] + list(samples)
        self.tumor_dat = tumor_data
        
        # 创建默认的细胞系数据
        cell_samples = ['ACH-000001', 'ACH-000002', 'ACH-000003']
        cell_data = pd.DataFrame(
            np.random.randn(len(genes), len(cell_samples)),
            index=genes,
            columns=cell_samples
        )
        cell_data = cell_data.reset_index()
        cell_data.columns = ['Gene'] + list(cell_samples)
        self.cell_dat = cell_data
        
        # 创建默认的HUGO数据
        self.hugo = pd.DataFrame({
            'symbol': genes,
            'name': [f'Gene {i}' for i in range(1, 6)],
            'ensembl_gene_id': [f'ENSG00000{i:06d}' for i in range(1, 6)],
            'location': ['chr1:1000-2000'] * 5,
            'locus_group': ['protein-coding gene'] * 5
        })
        
        # 创建默认的细胞依赖数据
        celldep_data = pd.DataFrame(
            np.random.randn(len(cell_samples), len(genes)),
            index=cell_samples,
            columns=genes
        )
        self.celldep = celldep_data
        
        # 创建默认的细胞信息数据
        self.cell_info = pd.DataFrame({
            'sampleID': cell_samples,
            'cell_line_name': [f'Cell_Line_{i}' for i in range(1, 4)],
            'tissue': ['Lung'] * 3,
            'cancer_type': ['NSCLC'] * 3
        })
    
    def _load_shared_data(self):
        """加载共享数据"""
        try:
            print("加载肿瘤表达数据...")
            self.tumor_dat = pd.read_csv(self.tumor_data_path, sep='\t')
            print(f"  肿瘤数据形状: {self.tumor_dat.shape}")
            
            print("加载细胞系表达数据...")
            self.cell_dat = pd.read_csv(self.cell_data_path)
            print(f"  细胞系数据形状: {self.cell_dat.shape}")
            
            print("加载HUGO基因注释...")
            self.hugo = pd.read_csv(self.hugo_path, sep='\t')
            print(f"  HUGO数据形状: {self.hugo.shape}")
            
            print("加载细胞依赖数据...")
            self.celldep = pd.read_csv(self.celldep_path)
            print(f"  细胞依赖数据形状: {self.celldep.shape}")
            
            print("加载细胞信息数据...")
            self.cell_info = pd.read_csv(self.cell_info_path)
            print(f"  细胞信息数据形状: {self.cell_info.shape}")
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            self._create_default_data()
    
    def _tidy_expression_data(self):
        """整理表达数据"""
        print("整理表达数据...")
        
        # 整理细胞系表达矩阵
        print("  整理细胞系表达数据...")
        cell_dat = self.cell_dat.copy()
        
        # 设置第一列为基因名索引
        if 'V1' in cell_dat.columns:
            cell_dat = cell_dat.set_index('V1')
        elif 'Gene' in cell_dat.columns:
            cell_dat = cell_dat.set_index('Gene')
        
        # 处理列名，取空格前的部分
        cell_dat.columns = [col.split(' ')[0] for col in cell_dat.columns]
        
        # 转置数据，使行为样本，列为基因
        cell_dat = cell_dat.T
        
        # 转换为DataFrame并添加Gene列
        cell_dat = pd.DataFrame(cell_dat)
        cell_dat = cell_dat.reset_index()
        cell_dat.columns = ['Gene'] + list(cell_dat.columns[1:])
        
        # 处理基因名中的点号
        cell_dat['Gene'] = cell_dat['Gene'].str.replace('.', '-')
        
        self.cell_dat = cell_dat
        print(f"  整理后细胞系数据形状: {self.cell_dat.shape}")
        
        # 整理肿瘤表达数据
        print("  整理肿瘤表达数据...")
        tumor_dat = self.tumor_dat.copy()
        
        # 设置Gene列为索引
        if 'Gene' in tumor_dat.columns:
            tumor_dat = tumor_dat.set_index('Gene')
        
        self.tumor_dat = tumor_dat
        print(f"  整理后肿瘤数据形状: {self.tumor_dat.shape}")
    
    def _filter_protein_coding_genes(self):
        """过滤蛋白质编码基因"""
        print("过滤蛋白质编码基因...")
        
        # 获取蛋白质编码基因
        protein_gene = self.hugo[self.hugo['locus_group'] == 'protein-coding gene'].copy()
        print(f"  蛋白质编码基因数量: {len(protein_gene)}")
        
        # 找到共同的基因
        tumor_genes = set(self.tumor_dat.index)
        cell_genes = set(self.cell_dat['Gene'])
        protein_genes = set(protein_gene['symbol'])
        
        sam_gene = tumor_genes.intersection(cell_genes).intersection(protein_genes)
        sam_gene = list(sam_gene)
        
        print(f"  共同基因数量: {len(sam_gene)}")
        
        # 过滤肿瘤数据
        self.tumor_dat = self.tumor_dat.loc[sam_gene]
        print(f"  过滤后肿瘤数据形状: {self.tumor_dat.shape}")
        
        # 过滤细胞系数据
        self.cell_dat = self.cell_dat[self.cell_dat['Gene'].isin(sam_gene)]
        self.cell_dat = self.cell_dat.set_index('Gene')
        print(f"  过滤后细胞系数据形状: {self.cell_dat.shape}")
    
    def _process_cell_dependency(self):
        """处理细胞依赖数据"""
        print("处理细胞依赖数据...")
        
        # 整理细胞依赖数据
        celldep = self.celldep.copy()
        
        # 设置第一列为索引
        if 'V1' in celldep.columns:
            celldep = celldep.set_index('V1')
        
        # 处理列名，取空格前的部分
        celldep.columns = [col.split(' ')[0] for col in celldep.columns]
        
        # 转换为DataFrame
        celldep = pd.DataFrame(celldep)
        
        # 找到共同的样本
        celldep_samples = set(celldep.index)
        cell_dat_samples = set(self.cell_dat.columns)
        sam_sample = list(celldep_samples.intersection(cell_dat_samples))
        
        print(f"  共同样本数量: {len(sam_sample)}")
        
        # 过滤细胞依赖数据
        self.celldep = celldep.loc[sam_sample]
        print(f"  过滤后细胞依赖数据形状: {self.celldep.shape}")
        
        # 过滤细胞系表达数据
        self.cell_dat = self.cell_dat[sam_sample]
        print(f"  过滤后细胞系表达数据形状: {self.cell_dat.shape}")
        
        # 过滤细胞信息数据
        self.cell_info = self.cell_info[self.cell_info['sampleID'].isin(sam_sample)]
        print(f"  过滤后细胞信息数据形状: {self.cell_info.shape}")
    
    def _create_nadata_object(self):
        """创建nadata对象"""
        print("创建nadata对象...")
        
        # 准备表达矩阵X (细胞系表达数据)
        X = self.cell_dat.values.T  # 转置，使行为样本，列为基因
        X = np.nan_to_num(X, nan=0.0)
        print(f"  表达矩阵X形状: {X.shape}")
        
        # 准备表型数据Meta (细胞依赖数据)
        Meta = self.celldep.reset_index()
        Meta.columns = ['id'] + list(Meta.columns[1:])
        print(f"  表型数据Meta形状: {Meta.shape}")
        
        # 准备基因数据Var
        genes = self.cell_dat.index.tolist()
        Var = self.hugo[self.hugo['symbol'].isin(genes)].copy()
        Var = Var[['symbol', 'name', 'ensembl_gene_id', 'location', 'locus_group']]
        Var.columns = ['Gene', 'Name', 'ENS', 'location', 'locus_group']
        print(f"  基因数据Var形状: {Var.shape}")
        
        # 确保数据一致性
        common_genes = set(genes).intersection(set(Var['Gene']))
        common_samples = list(Meta['id'])
        
        if len(common_genes) < 5:
            print(f"  警告: 基因数量不足 (当前: {len(common_genes)})")
            return None
        
        if len(common_samples) < 10:
            print(f"  警告: 样本数量不足 (当前: {len(common_samples)})")
            return None
        
        # 过滤数据
        X = self.cell_dat.loc[list(common_genes), common_samples].values.T
        X = np.nan_to_num(X, nan=0.0)
        
        Meta = Meta[Meta['id'].isin(common_samples)].copy()
        Var = Var[Var['Gene'].isin(common_genes)].copy()
        
        # 创建nadata对象
        nadata_obj = nadata(
            X=X,
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
        
        print(f"  nadata对象创建完成: 样本{len(common_samples)}, 基因{len(common_genes)}")
        return nadata_obj
    
    def _save_data(self, nadata_obj):
        """保存数据"""
        print("保存数据...")
        
        # 保存nadata对象
        nadata_path = self.output_dir / "cell_tumor_dependency.pkl"
        with open(nadata_path, 'wb') as f:
            pickle.dump(nadata_obj, f)
        print(f"  nadata对象已保存到: {nadata_path}")
        
        # 保存肿瘤数据为CSV
        tumor_csv_path = self.output_dir / "tumor_data.csv"
        self.tumor_dat.to_csv(tumor_csv_path)
        print(f"  肿瘤数据已保存到: {tumor_csv_path}")
        
        # 保存细胞信息数据为CSV
        cell_info_csv_path = self.output_dir / "cell_info.csv"
        self.cell_info.to_csv(cell_info_csv_path, index=False)
        print(f"  细胞信息数据已保存到: {cell_info_csv_path}")
        
        # 保存细胞系表达数据为CSV
        cell_exp_csv_path = self.output_dir / "cell_expression.csv"
        self.cell_dat.to_csv(cell_exp_csv_path)
        print(f"  细胞系表达数据已保存到: {cell_exp_csv_path}")
        
        # 保存细胞依赖数据为CSV
        celldep_csv_path = self.output_dir / "cell_dependency.csv"
        self.celldep.to_csv(celldep_csv_path)
        print(f"  细胞依赖数据已保存到: {celldep_csv_path}")
    
    def process(self):
        """主处理函数"""
        print("开始处理细胞系-肿瘤依赖关系数据...")
        
        start_time = time.time()
        
        try:
            # 整理表达数据
            self._tidy_expression_data()
            
            # 过滤蛋白质编码基因
            self._filter_protein_coding_genes()
            
            # 处理细胞依赖数据
            self._process_cell_dependency()
            
            # 创建nadata对象
            nadata_obj = self._create_nadata_object()
            
            if nadata_obj is not None:
                # 保存数据
                self._save_data(nadata_obj)
                
                end_time = time.time()
                print(f"\n处理完成!")
                print(f"总耗时: {end_time - start_time:.2f} 秒")
                print(f"数据已保存到: {self.output_dir}")
                
                return nadata_obj
            else:
                print("创建nadata对象失败")
                return None
                
        except Exception as e:
            print(f"处理过程中出错: {e}")
            return None

def main():
    """主函数"""
    processor = CellTumorDependencyProcessor()
    nadata_obj = processor.process()
    
    if nadata_obj is not None:
        print("\n数据摘要:")
        print(f"  表达矩阵X形状: {nadata_obj.X.shape}")
        print(f"  表型数据Meta形状: {nadata_obj.Meta.shape}")
        print(f"  基因数据Var形状: {nadata_obj.Var.shape}")
        print(f"  训练样本数: {len(nadata_obj.Model.get_indices('train'))}")
        print(f"  验证样本数: {len(nadata_obj.Model.get_indices('val'))}")
        print(f"  测试样本数: {len(nadata_obj.Model.get_indices('test'))}")

if __name__ == "__main__":
    main()
