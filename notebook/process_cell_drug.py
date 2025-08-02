#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
细胞系药物数据处理脚本
将R脚本转换为Python版本，处理GDSC数据库中的细胞系药物数据
为每个药物创建nadata对象，保存到datasets/cell_drug文件夹中
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

class CellDrugProcessor:
    """细胞系药物数据处理器"""
    
    def __init__(self):
        """
        初始化处理器
        """
        self.results = {}
        
        # 数据文件路径 - 使用绝对路径
        base_dir = Path(__file__).parent
        self.drug_sensitivity_path = base_dir / "data" / "cell_line" / "Drug_sensitivity_AUC_(Sanger_GDSC2)_subsetted.csv"
        self.expression_path = base_dir / "data" / "cell_line" / "cell_exp.csv"
        self.hugo_geneset_path = base_dir / "data" / "hugo_2025.txt"
        
        # 检查文件是否存在
        self._check_files()
        
        # 加载共享数据
        self._load_shared_data()
    
    def _check_files(self):
        """检查必要文件是否存在"""
        required_files = [
            self.drug_sensitivity_path,
            self.expression_path,
            self.hugo_geneset_path
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
        
        # 创建默认的药物敏感性数据
        self.drug_sensitivity = pd.DataFrame({
            'cell_id': ['ACH-000001', 'ACH-000002', 'ACH-000003'],
            'LAPATINIB (GDSC2:1558)': [0.8, 0.6, 0.9],
            'BORTEZOMIB (GDSC2:1191)': [0.7, 0.5, 0.8],
            'CYTARABINE (GDSC2:1006)': [0.6, 0.4, 0.7]
        })
        
        # 创建默认的表达数据
        genes = ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5']
        cell_ids = ['ACH-000001', 'ACH-000002', 'ACH-000003']
        
        np.random.seed(42)
        exp_data = pd.DataFrame(
            np.random.randn(len(genes), len(cell_ids)),
            index=genes,
            columns=cell_ids
        )
        exp_data.insert(0, 'Gene', genes)
        self.expression_data = exp_data
        
        # 创建默认的HUGO基因集
        self.hugo_geneset = pd.DataFrame({
            'symbol': genes,
            'name': [f'Gene {i}' for i in range(1, len(genes)+1)],
            'ensembl_gene_id': [f'ENSG00000{i}' for i in range(1, len(genes)+1)],
            'location': [f'chr{i}' for i in range(1, len(genes)+1)],
            'locus_group': ['protein-coding gene'] * len(genes)
        })
    
    def _load_shared_data(self):
        """加载共享数据"""
        print("加载共享数据...")
        
        try:
            # 加载药物敏感性数据
            print("  加载药物敏感性数据...")
            if self.drug_sensitivity_path.exists():
                try:
                    self.drug_sensitivity = pd.read_csv(self.drug_sensitivity_path, index_col=0)
                    print(f"  药物敏感性数据形状: {self.drug_sensitivity.shape}")
                    print(f"  药物数量: {len(self.drug_sensitivity.columns)}")
                    print(f"  细胞系数量: {len(self.drug_sensitivity.index)}")
                except Exception as e:
                    print(f"  加载药物敏感性数据失败: {e}")
                    self._create_default_data()
            else:
                print(f"  药物敏感性文件不存在: {self.drug_sensitivity_path}")
                self._create_default_data()
            
            # 加载表达数据
            print("  加载表达数据...")
            if self.expression_path.exists():
                try:
                    # 由于文件很大，只读取前几行来了解结构
                    self.expression_data = pd.read_csv(self.expression_path)
                    print(f"  表达数据形状: {self.expression_data.shape}")
                    print(f"  基因数量: {len(self.expression_data)}")
                    print(f"  样本数量: {len(self.expression_data.columns) - 1}")  # 减去Gene列
                except Exception as e:
                    print(f"  加载表达数据失败: {e}")
                    self._create_default_data()
            else:
                print(f"  表达数据文件不存在: {self.expression_path}")
                self._create_default_data()
            
            # 加载HUGO基因集
            print("  加载HUGO基因集...")
            if self.hugo_geneset_path.exists():
                try:
                    self.hugo_geneset = pd.read_csv(self.hugo_geneset_path, sep='\t')
                    print(f"  HUGO基因集形状: {self.hugo_geneset.shape}")
                except Exception as e:
                    print(f"  加载HUGO基因集失败: {e}")
                    self._create_default_data()
            else:
                print(f"  HUGO基因集文件不存在: {self.hugo_geneset_path}")
                self._create_default_data()
            
            # 获取所有可用的药物
            print("  分析药物类型...")
            self.drug_list = self._get_drug_list()
            print(f"  发现 {len(self.drug_list)} 种药物")
            
        except Exception as e:
            print(f"加载共享数据时出错: {e}")
            import traceback
            traceback.print_exc()
            self._create_default_data()
    
    def _get_drug_list(self):
        """获取所有药物列表"""
        try:
            # 从tumor_drug中获取药物列表
            base_dir = Path(__file__).parent
            tumor_drug_dir = base_dir / "datasets" / "tumor_drug"
            
            if tumor_drug_dir.exists():
                pkl_files = [f for f in tumor_drug_dir.iterdir() if f.name.endswith('_drug.pkl')]
                drugs = []
                for pkl_file in pkl_files:
                    drug_name = pkl_file.stem.replace('_drug', '')
                    drugs.append(drug_name)
                
                # 过滤掉空值
                drugs = [drug for drug in drugs if drug and str(drug).strip()]
                
                print(f"  从tumor_drug中找到 {len(drugs)} 种药物")
                return drugs
            else:
                print("  tumor_drug目录不存在，使用默认药物列表")
                return ['LAPATINIB', 'BORTEZOMIB', 'CYTARABINE']
        except Exception as e:
            print(f"获取药物列表时出错: {e}")
            return ['LAPATINIB', 'BORTEZOMIB', 'CYTARABINE']  # 返回默认药物列表
    
    def process_single_drug(self, drug_name):
        """
        处理单个药物的数据
        
        Args:
            drug_name: 药物名称
            
        Returns:
            tuple: (drug_name, success, nadata_obj or error_message)
        """
        try:
            print(f"开始处理 {drug_name}...")
            
            # 在药物敏感性数据中查找包含该药物的列
            # 使用正则表达式匹配药物名称（不区分大小写）
            drug_pattern = re.compile(re.escape(drug_name), re.IGNORECASE)
            
            # 查找包含该药物的列
            drug_columns = []
            for col in self.drug_sensitivity.columns:
                if drug_pattern.search(col):
                    drug_columns.append(col)
            
            if not drug_columns:
                print(f"  {drug_name}: 未找到包含该药物的列")
                return drug_name, False, f"未找到包含 {drug_name} 的列"
            
            # 使用第一个匹配的列
            drug_column = drug_columns[0]
            print(f"  使用药物列: {drug_column}")
            
            # 提取药物敏感性数据
            drug_data = self.drug_sensitivity[[drug_column]].copy()
            drug_data.columns = ['auc']
            drug_data = drug_data.dropna()  # 移除缺失值
            
            if len(drug_data) < 20:
                print(f"  {drug_name}: 样本数量不足20个，跳过处理 (当前样本数: {len(drug_data)})")
                return drug_name, False, f"样本数量不足20个: {len(drug_data)}"
            
            print(f"  找到 {len(drug_data)} 个包含 {drug_name} 的样本")
            
            # 获取细胞系ID
            cell_ids = drug_data.index.tolist()
            
            # 从表达数据中提取对应的细胞系数据
            # 假设表达数据的第一列是基因名，其余列是细胞系
            exp_data = self.expression_data.copy()
            
            # 设置基因列为索引
            if 'Gene' in exp_data.columns:
                exp_data = exp_data.set_index('Gene')
            
            # 选择共同基因
            hugo_gene = self.hugo_geneset[self.hugo_geneset['locus_group'] == 'protein-coding gene'].copy()
            if len(hugo_gene) == 0:
                # 如果没有protein-coding gene，使用所有基因
                hugo_gene = self.hugo_geneset.copy()
            
            # 获取基因列表
            genes = hugo_gene['symbol'].tolist()
            
            # 过滤表达数据，只保留有基因数据的行
            available_genes = [gene for gene in genes if gene in exp_data.index]
            
            if len(available_genes) < 10:
                print(f"  {drug_name}: 基因数量不足 (当前: {len(available_genes)})")
                return drug_name, False, f"基因数量不足: {len(available_genes)}"
            
            # 过滤表达数据，只保留有细胞系数据的列
            available_cells = [cell_id for cell_id in cell_ids if cell_id in exp_data.columns]
            
            if len(available_cells) < 20:
                print(f"  {drug_name}: 细胞系数量不足20个 (当前: {len(available_cells)})")
                return drug_name, False, f"细胞系数量不足20个: {len(available_cells)}"
            
            # 准备表达矩阵X
            X = exp_data.loc[available_genes, available_cells].values.T  # 转置，使行为样本，列为基因
            
            # 处理缺失值
            X = np.nan_to_num(X, nan=0.0)
            
            # 构建表型数据
            Meta = pd.DataFrame({
                'id': available_cells,
                'cell_id': available_cells,
                'auc': [drug_data.loc[cell_id, 'auc'] for cell_id in available_cells if cell_id in drug_data.index]
            })
            
            # 过滤表型数据，只保留有表达数据的细胞系
            Meta = Meta[Meta['id'].isin(available_cells)].copy()
            
            # 构建基因数据
            Var = hugo_gene[['symbol', 'name', 'ensembl_gene_id', 'location', 'locus_group']].copy()
            Var.columns = ['Gene', 'Name', 'ENS', 'location', 'locus_group']
            
            # 过滤基因数据，只保留表达矩阵中的基因
            Var = Var[Var['Gene'].isin(available_genes)].copy()
            
            # 确保数据一致性
            common_genes_final = set(available_genes).intersection(set(Var['Gene']))
            common_cells = list(Meta['id'])
            
            if len(common_genes_final) < 5:
                print(f"  {drug_name}: 最终基因数量不足 (当前: {len(common_genes_final)})")
                return drug_name, False, f"最终基因数量不足: {len(common_genes_final)}"
            
            if len(common_cells) < 20:
                print(f"  {drug_name}: 最终细胞系数量不足20个 (当前: {len(common_cells)})")
                return drug_name, False, f"最终细胞系数量不足20个: {len(common_cells)}"
            
            # 准备表达矩阵X
            X = exp_data.loc[list(common_genes_final), common_cells].values.T  # 转置，使行为样本，列为基因
            X = np.nan_to_num(X, nan=0.0)
            
            # 过滤表型数据
            Meta = Meta[Meta['id'].isin(common_cells)].copy()
            
            # 过滤基因数据
            Var = Var[Var['Gene'].isin(common_genes_final)].copy()
            
            # 创建nadata对象
            nadata_obj = nadata(
                X=X,
                Meta=Meta,
                Var=Var,
                Prior=None
            )
            
            # 设置训练/测试/验证索引
            n_samples = len(common_cells)
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
            
            print(f"  {drug_name} 处理完成: 细胞系{len(common_cells)}, 基因{len(common_genes_final)}")
            return drug_name, True, nadata_obj
            
        except Exception as e:
            error_msg = f"处理 {drug_name} 时出错: {str(e)}"
            print(f"  {error_msg}")
            return drug_name, False, error_msg
    
    def process_all_drugs(self):
        """单线程处理所有药物"""
        print(f"开始单线程处理 {len(self.drug_list)} 种药物...")
        
        start_time = time.time()
        
        completed = 0
        successful = 0
        failed_reasons = defaultdict(int)  # 统计失败原因
        
        # 逐个处理每个药物
        for drug_name in self.drug_list:
            try:
                drug_name, success, result = self.process_single_drug(drug_name)
                completed += 1
                
                if success:
                    successful += 1
                    # 保存数据
                    self._save_drug_data(drug_name, result)
                else:
                    # 统计失败原因
                    failed_reasons[result] += 1
                
                print(f"进度: {completed}/{len(self.drug_list)} ({successful} 成功)")
                
            except Exception as e:
                completed += 1
                error_msg = f"处理异常: {str(e)}"
                failed_reasons[error_msg] += 1
                print(f"处理 {drug_name} 时发生异常: {e}")
        
        end_time = time.time()
        print(f"\n处理完成!")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"成功处理: {successful}/{len(self.drug_list)} 种药物")
        
        # 打印失败原因统计
        if failed_reasons:
            print(f"\n失败原因统计:")
            for reason, count in failed_reasons.items():
                print(f"  {reason}: {count} 个药物")
    
    def _save_drug_data(self, drug_name, nadata_obj):
        """保存药物数据"""
        try:
            # 创建安全的文件名
            safe_name = drug_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
            
            # 使用绝对路径
            base_dir = Path(__file__).parent
            output_dir = base_dir / "datasets" / "cell_drug"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为pickle格式
            output_path = output_dir / f"{safe_name}_drug.pkl"
            
            try:
                nadata_obj.save(str(output_path), format='pickle', save_data=True)
                print(f"  数据已保存到: {output_path}")
                
                # 记录成功信息
                self.results[safe_name] = {
                    'drug_name': drug_name,
                    'file_path': str(output_path),
                    'shape': nadata_obj.X.shape,
                    'meta_shape': nadata_obj.Meta.shape,
                    'var_shape': nadata_obj.Var.shape
                }
                    
            except Exception as e:
                print(f"  保存 {drug_name} 数据时出错: {e}")
                # 尝试使用pickle直接保存
                try:
                    with open(output_path, 'wb') as f:
                        pickle.dump(nadata_obj, f)
                    print(f"  使用pickle直接保存成功: {output_path}")
                    
                    self.results[safe_name] = {
                        'drug_name': drug_name,
                        'file_path': str(output_path),
                        'shape': nadata_obj.X.shape,
                        'meta_shape': nadata_obj.Meta.shape,
                        'var_shape': nadata_obj.Var.shape
                    }
                except Exception as e2:
                    print(f"  pickle保存也失败: {e2}")
                    
        except Exception as e:
            print(f"  保存 {drug_name} 数据时出错: {e}")
    
    def get_processing_summary(self):
        """获取处理摘要"""
        print("\n处理摘要:")
        print(f"发现药物类型: {len(self.drug_list)}")
        
        for drug in self.drug_list:
            print(f"  {drug}")
        
        print(f"\n成功处理: {len(self.results)}")
        for safe_name, info in self.results.items():
            print(f"  {info['drug_name']}: {info['shape']}")
        
        # 计算失败数量
        failed_count = len(self.drug_list) - len(self.results)
        if failed_count > 0:
            print(f"\n失败处理: {failed_count}")
            print("失败原因可能包括:")
            print("  - 样本数量不足20个")
            print("  - 未找到包含该药物的列")
            print("  - 基因数量不足")
            print("  - 细胞系数量不足20个")
            print("  - 最终基因数量不足")
            print("  - 最终细胞系数量不足20个")

def load_cell_drug_data(drug_name=None):
    """
    加载细胞系药物数据
    
    Args:
        drug_name: 药物名称，如果为None则返回所有可用的药物数据
        
    Returns:
        dict: 药物数据字典
    """
    # 使用绝对路径
    base_dir = Path(__file__).parent
    output_dir = base_dir / "datasets" / "cell_drug"
    
    if not output_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {output_dir}")
    
    # 查找所有pkl文件
    pkl_files = [f for f in output_dir.iterdir() if f.name.endswith('_drug.pkl')]
    
    if not pkl_files:
        raise FileNotFoundError(f"在 {output_dir} 中未找到任何细胞系药物数据文件")
    
    results = {}
    
    for pkl_file in pkl_files:
        drug_name_file = pkl_file.stem.replace('_drug', '')
        
        if drug_name is None or drug_name.lower() in drug_name_file.lower():
            try:
                print(f"加载 {drug_name_file} 数据...")
                
                # 尝试使用nadata的load方法
                try:
                    nadata_obj = nadata()
                    nadata_obj.load(str(pkl_file))
                except:
                    # 如果失败，尝试直接pickle加载
                    with open(pkl_file, 'rb') as f:
                        nadata_obj = pickle.load(f)
                
                results[drug_name_file] = {
                    'data': nadata_obj,
                    'file_path': str(pkl_file),
                    'shape': nadata_obj.X.shape,
                    'meta_shape': nadata_obj.Meta.shape,
                    'var_shape': nadata_obj.Var.shape
                }
                
                print(f"  {drug_name_file} 加载成功: {nadata_obj.X.shape}")
                
            except Exception as e:
                print(f"  加载 {drug_name_file} 时出错: {e}")
    
    return results

if __name__ == "__main__":
    try:
        print("=" * 50)
        print("细胞系药物数据处理程序开始运行")
        print("=" * 50)
        
        # 创建处理器并处理所有药物
        processor = CellDrugProcessor()
        
        # 显示处理摘要
        processor.get_processing_summary()
        
        # 开始处理
        processor.process_all_drugs()
        
        # 测试加载
        print("\n测试数据加载...")
        try:
            loaded_data = load_cell_drug_data()
            print(f"成功加载 {len(loaded_data)} 个药物数据集")
        except Exception as e:
            print(f"加载数据时出错: {e}")
        
        print("\n数据处理完成!")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()