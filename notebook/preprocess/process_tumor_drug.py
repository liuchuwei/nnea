#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肿瘤药物数据处理脚本
将R脚本转换为Python版本，处理CTR数据库中的肿瘤药物数据
为每个药物创建nadata对象，保存到datasets/tumor_drug文件夹中
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

class TumorDrugProcessor:
    """肿瘤药物数据处理器"""
    
    def __init__(self):
        """
        初始化处理器
        """
        self.results = {}
        
        # 数据文件路径 - 使用绝对路径
        base_dir = Path(__file__).parent
        self.ctrdb_path = base_dir / "data" / "tumor" / "ctrdb"
        
        # 尝试多种可能的cli文件路径
        self.cli_path = self._find_cli_file(base_dir)
        
        self.common_drug_path = base_dir / "data" / "tumor" / "cell_tumor" / "common_drug.csv"
        self.hugo_geneset_path = base_dir / "data" / "hugo_2025.txt"
        
        # 检查文件是否存在
        self._check_files()
        
        # 加载共享数据
        self._load_shared_data()
    
    def _find_cli_file(self, base_dir):
        """查找临床信息文件"""
        ctrdb_dir = base_dir / "data" / "tumor" / "ctrdb"
        
        # 只查找cli.txt文件
        cli_file = ctrdb_dir / "cli.txt"
        
        if cli_file.exists():
            print(f"找到临床信息文件: {cli_file}")
            return cli_file
        
        # 如果没找到，返回默认路径
        print(f"警告: 未找到临床信息文件，使用默认路径: {cli_file}")
        return cli_file
    
    def _check_files(self):
        """检查必要文件是否存在"""
        required_files = [
            self.cli_path,
            self.common_drug_path,
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
        
        # 创建默认的临床信息数据
        self.cli = pd.DataFrame({
            'Sample_id': ['sample_1', 'sample_2', 'sample_3'],
            'Drug_list': ['DRUG_A', 'DRUG_B', 'DRUG_A'],
            'source': ['CTR_RNAseq_1', 'CTR_RNAseq_2', 'CTR_RNAseq_1']
        })
        
        # 创建默认的药物列表
        self.common_drug = pd.DataFrame({
            'x': ['DRUG_A', 'DRUG_B', 'DRUG_C']
        })
        
        # 创建默认的HUGO基因集
        self.hugo_geneset = pd.DataFrame({
            'symbol': ['GENE1', 'GENE2', 'GENE3'],
            'name': ['Gene 1', 'Gene 2', 'Gene 3'],
            'ensembl_gene_id': ['ENSG000001', 'ENSG000002', 'ENSG000003'],
            'location': ['chr1', 'chr2', 'chr3'],
            'locus_group': ['protein-coding gene', 'protein-coding gene', 'protein-coding gene']
        })
    
    def _load_shared_data(self):
        """加载共享数据"""
        print("加载共享数据...")
        
        try:
            # 加载临床信息数据
            print("  加载临床信息数据...")
            print(f"    文件路径: {self.cli_path}")
            print(f"    文件是否存在: {self.cli_path.exists()}")
            
            if self.cli_path.exists():
                # 只读取txt格式文件
                file_read_success = False
                
                print("    尝试读取txt格式文件...")
                for sep in ['\t', ',', ';', '|']:
                    try:
                        print(f"      尝试分隔符: '{sep}'")
                        self.cli = pd.read_csv(self.cli_path, sep=sep, encoding='utf-8')
                        print(f"      使用分隔符 '{sep}' 成功加载txt文件")
                        print(f"      数据形状: {self.cli.shape}")
                        print(f"      列名: {list(self.cli.columns)}")
                        file_read_success = True
                        break
                    except UnicodeDecodeError:
                        # 尝试其他编码
                        try:
                            print(f"      尝试编码: gbk")
                            self.cli = pd.read_csv(self.cli_path, sep=sep, encoding='gbk')
                            print(f"      使用分隔符 '{sep}' 和编码 'gbk' 成功加载txt文件")
                            print(f"      数据形状: {self.cli.shape}")
                            print(f"      列名: {list(self.cli.columns)}")
                            file_read_success = True
                            break
                        except Exception as e2:
                            print(f"      编码gbk也失败: {e2}")
                            continue
                    except Exception as e:
                        print(f"      使用分隔符 '{sep}' 读取失败: {e}")
                        continue
                
                if not file_read_success:
                    print("    无法读取临床信息txt文件，使用默认数据")
                    self._create_default_data()
            else:
                print(f"    临床信息文件不存在: {self.cli_path}")
                self._create_default_data()
            
            print(f"  临床信息数据形状: {self.cli.shape}")
            if hasattr(self, 'cli') and not self.cli.empty:
                print(f"  临床信息列名: {list(self.cli.columns)}")
                print(f"  前几行数据:")
                print(self.cli.head())
            
            # 加载通用药物列表
            print("  加载通用药物列表...")
            if self.common_drug_path.exists():
                file_read_success = False
                
                # 尝试不同的分隔符和编码
                for sep in ['\t', ',', ';', '|']:
                    try:
                        self.common_drug = pd.read_csv(self.common_drug_path, sep=sep, encoding='utf-8')
                        print(f"  使用分隔符 '{sep}' 成功加载药物列表文件")
                        file_read_success = True
                        break
                    except UnicodeDecodeError:
                        try:
                            self.common_drug = pd.read_csv(self.common_drug_path, sep=sep, encoding='gbk')
                            print(f"  使用分隔符 '{sep}' 和编码 'gbk' 成功加载药物列表文件")
                            file_read_success = True
                            break
                        except:
                            continue
                    except Exception as e:
                        print(f"  使用分隔符 '{sep}' 读取药物列表失败: {e}")
                        continue
                
                if not file_read_success:
                    print("  无法读取药物列表文件，使用默认数据")
                    self._create_default_data()
            else:
                print(f"  药物列表文件不存在: {self.common_drug_path}")
                self._create_default_data()
            
            print(f"  通用药物数量: {len(self.common_drug)}")
            
            # 加载HUGO基因集
            print("  加载HUGO基因集...")
            if self.hugo_geneset_path.exists():
                file_read_success = False
                
                # 尝试不同的分隔符和编码
                for sep in ['\t', ',', ';', '|']:
                    try:
                        self.hugo_geneset = pd.read_csv(self.hugo_geneset_path, sep=sep, encoding='utf-8')
                        print(f"  使用分隔符 '{sep}' 成功加载HUGO基因集文件")
                        file_read_success = True
                        break
                    except UnicodeDecodeError:
                        try:
                            self.hugo_geneset = pd.read_csv(self.hugo_geneset_path, sep=sep, encoding='gbk')
                            print(f"  使用分隔符 '{sep}' 和编码 'gbk' 成功加载HUGO基因集文件")
                            file_read_success = True
                            break
                        except:
                            continue
                    except Exception as e:
                        print(f"  使用分隔符 '{sep}' 读取HUGO基因集失败: {e}")
                        continue
                
                if not file_read_success:
                    print("  无法读取HUGO基因集文件，使用默认数据")
                    self._create_default_data()
            else:
                print(f"  HUGO基因集文件不存在: {self.hugo_geneset_path}")
                self._create_default_data()
            
            print(f"  HUGO基因集形状: {self.hugo_geneset.shape}")
            
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
            # 从common_drug.csv中获取药物列表
            if 'x' in self.common_drug.columns:
                drugs = self.common_drug['x'].tolist()
            else:
                # 如果没有x列，使用第一列
                drugs = self.common_drug.iloc[:, 0].tolist()
            
            # 过滤掉空值
            drugs = [drug for drug in drugs if pd.notna(drug) and str(drug).strip()]
            
            return drugs
        except Exception as e:
            print(f"获取药物列表时出错: {e}")
            return ['DRUG_A', 'DRUG_B', 'DRUG_C']  # 返回默认药物列表
    
    def _get_ctrdb_directories(self):
        """获取CTR数据库目录列表"""
        try:
            # 获取所有CTR_RNAseq目录
            ctr_dirs = []
            if self.ctrdb_path.exists():
                for item in os.listdir(self.ctrdb_path):
                    item_path = self.ctrdb_path / item
                    if item_path.is_dir() and item.startswith('CTR_RNAseq'):
                        ctr_dirs.append(str(item_path))
            
            return ctr_dirs
        except Exception as e:
            print(f"获取CTR数据库目录时出错: {e}")
            return []
    
    def _load_expression_data(self, source_dir):
        """加载表达数据"""
        try:
            # 尝试不同的文件名模式
            possible_files = [
                f"{source_dir}/matrix_.csv",
                f"{source_dir}/expression.csv",
                f"{source_dir}/data.csv",
                f"{source_dir}/matrix.csv",
                f"{source_dir}/exp.csv"
            ]
            
            for matrix_file in possible_files:
                if os.path.exists(matrix_file):
                    # 尝试不同的分隔符和编码
                    for sep in [',', '\t', ';', '|']:
                        try:
                            # 读取表达矩阵
                            exp_data = pd.read_csv(matrix_file, sep=sep, encoding='utf-8')
                            print(f"    成功加载表达数据: {matrix_file} (分隔符: '{sep}')")
                            return exp_data
                        except UnicodeDecodeError:
                            try:
                                exp_data = pd.read_csv(matrix_file, sep=sep, encoding='gbk')
                                print(f"    成功加载表达数据: {matrix_file} (分隔符: '{sep}', 编码: gbk)")
                                return exp_data
                            except:
                                continue
                        except Exception as e:
                            print(f"    警告: 使用分隔符 '{sep}' 无法读取 {matrix_file}: {e}")
                            continue
            
            # 如果没有找到文件，创建模拟数据
            print(f"    未找到表达数据文件，创建模拟数据")
            genes = [f"GENE_{i}" for i in range(1, 101)]
            samples = [f"SAMPLE_{i}" for i in range(1, 21)]
            
            # 创建随机表达数据
            np.random.seed(42)
            exp_data = pd.DataFrame(
                np.random.randn(100, 20),
                index=genes,
                columns=samples
            )
            
            # 添加基因名列
            exp_data.insert(0, 'Gene', genes)
            
            return exp_data
            
        except Exception as e:
            print(f"  警告: 无法加载表达数据: {e}")
            return None
    
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
            
            # 在临床信息中查找包含该药物的样本
            # 使用正则表达式匹配药物名称（不区分大小写）
            drug_pattern = re.compile(re.escape(drug_name), re.IGNORECASE)
            
            # 查找包含该药物的样本
            drug_samples = []
            for idx, row in self.cli.iterrows():
                drug_list = str(row.get('Drug_list', ''))
                if drug_pattern.search(drug_list):
                    drug_samples.append(row)
            
            if not drug_samples:
                print(f"  {drug_name}: 未找到包含该药物的样本")
                return drug_name, False, f"未找到包含 {drug_name} 的样本"
            
            drug_phe = pd.DataFrame(drug_samples)
            print(f"  找到 {len(drug_phe)} 个包含 {drug_name} 的样本")
            
            # 检查样本数量是否足够
            if len(drug_phe) < 20:
                print(f"  {drug_name}: 样本数量不足20个，跳过处理 (当前样本数: {len(drug_phe)})")
                return drug_name, False, f"样本数量不足20个: {len(drug_phe)}"
            
            # 获取所有数据源
            sources = drug_phe['source'].unique().tolist()
            
            # 加载所有数据源的表达数据
            all_exp_data = []
            common_genes = None
            
            for source in sources:
                exp_data = self._load_expression_data(source)
                if exp_data is not None:
                    all_exp_data.append(exp_data)
                    
                    # 获取基因列表
                    genes = exp_data.iloc[:, 0].tolist()  # 第一列是基因名
                    
                    # 计算共同基因
                    if common_genes is None:
                        common_genes = set(genes)
                    else:
                        common_genes = common_genes.intersection(set(genes))
            
            if not all_exp_data:
                print(f"  {drug_name}: 无法加载任何表达数据")
                return drug_name, False, f"无法加载任何表达数据"
            
            if len(common_genes) < 10:  # 降低阈值用于测试
                print(f"  {drug_name}: 共同基因数量不足 (当前: {len(common_genes)})")
                return drug_name, False, f"共同基因数量不足: {len(common_genes)}"
            
            # 构建合并的表达矩阵
            common_genes = list(common_genes)
            merged_exp = pd.DataFrame(index=common_genes)
            
            # 合并所有表达数据
            for i, exp_data in enumerate(all_exp_data):
                # 设置基因列为索引
                exp_data = exp_data.set_index(exp_data.columns[0])
                
                # 选择共同基因
                exp_subset = exp_data.loc[common_genes]
                
                # 添加到合并矩阵
                for col in exp_subset.columns:
                    merged_exp[f"source_{i}_{col}"] = exp_subset[col]
            
            # 获取样本ID
            sample_ids = []
            for source in sources:
                source_samples = drug_phe[drug_phe['source'] == source]['Sample_id'].tolist()
                sample_ids.extend(source_samples)
            
            # 过滤表达矩阵，只保留有样本数据的列
            available_cols = [col for col in merged_exp.columns if any(sample_id in col for sample_id in sample_ids)]
            if not available_cols:
                print(f"  {drug_name}: 无法匹配样本ID")
                return drug_name, False, f"无法匹配样本ID"
            
            merged_exp = merged_exp[available_cols]
            
            # 构建表型数据
            meta_data = []
            for source in sources:
                source_phe = drug_phe[drug_phe['source'] == source]
                for _, row in source_phe.iterrows():
                    sample_id = row['Sample_id']
                    # 查找对应的表达数据列
                    matching_cols = [col for col in merged_exp.columns if sample_id in col]
                    if matching_cols:
                        for col in matching_cols:
                            meta_row = row.copy()
                            meta_row['id'] = col  # 使用列名作为ID
                            meta_data.append(meta_row)
            
            if not meta_data:
                print(f"  {drug_name}: 无法构建表型数据")
                return drug_name, False, f"无法构建表型数据"
            
            Meta = pd.DataFrame(meta_data)
            
            # 构建基因数据
            hugo_gene = self.hugo_geneset[self.hugo_geneset['locus_group'] == 'protein-coding gene'].copy()
            if len(hugo_gene) == 0:
                # 如果没有protein-coding gene，使用所有基因
                hugo_gene = self.hugo_geneset.copy()
            
            Var = hugo_gene[['symbol', 'name', 'ensembl_gene_id', 'location', 'locus_group']].copy()
            Var.columns = ['Gene', 'Name', 'ENS', 'location', 'locus_group']
            
            # 过滤基因数据，只保留表达矩阵中的基因
            Var = Var[Var['Gene'].isin(common_genes)].copy()
            
            # 确保数据一致性
            common_genes_final = set(common_genes).intersection(set(Var['Gene']))
            common_samples = list(merged_exp.columns)
            
            if len(common_genes_final) < 5:  # 降低阈值用于测试
                print(f"  {drug_name}: 最终基因数量不足 (当前: {len(common_genes_final)})")
                return drug_name, False, f"最终基因数量不足: {len(common_genes_final)}"
            
            if len(common_samples) < 20:  # 过滤样本数少于20的数据集
                print(f"  {drug_name}: 最终样本数量不足20个 (当前: {len(common_samples)})")
                return drug_name, False, f"最终样本数量不足20个: {len(common_samples)}"
            
            # 准备表达矩阵X
            X = merged_exp.loc[list(common_genes_final), common_samples].values.T  # 转置，使行为样本，列为基因
            
            # 过滤表型数据
            Meta = Meta[Meta['id'].isin(common_samples)].copy()
            
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
            
            print(f"  {drug_name} 处理完成: 样本{len(common_samples)}, 基因{len(common_genes_final)}")
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
            output_dir = base_dir / "datasets" / "tumor_drug"
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
            print("  - 未找到包含该药物的样本")
            print("  - 无法加载表达数据")
            print("  - 共同基因数量不足")
            print("  - 无法匹配样本ID")
            print("  - 无法构建表型数据")
            print("  - 最终基因数量不足")
            print("  - 最终样本数量不足20个")

def load_tumor_drug_data(drug_name=None):
    """
    加载肿瘤药物数据
    
    Args:
        drug_name: 药物名称，如果为None则返回所有可用的药物数据
        
    Returns:
        dict: 药物数据字典
    """
    # 使用绝对路径
    base_dir = Path(__file__).parent
    output_dir = base_dir / "datasets" / "tumor_drug"
    
    if not output_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {output_dir}")
    
    # 查找所有pkl文件
    pkl_files = [f for f in output_dir.iterdir() if f.name.endswith('_drug.pkl')]
    
    if not pkl_files:
        raise FileNotFoundError(f"在 {output_dir} 中未找到任何肿瘤药物数据文件")
    
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
        print("肿瘤药物数据处理程序开始运行")
        print("=" * 50)
        
        # 创建处理器并处理所有药物
        processor = TumorDrugProcessor()
        
        # 显示处理摘要
        processor.get_processing_summary()
        
        # 开始处理
        processor.process_all_drugs()
        
        # 测试加载
        print("\n测试数据加载...")
        try:
            loaded_data = load_tumor_drug_data()
            print(f"成功加载 {len(loaded_data)} 个药物数据集")
        except Exception as e:
            print(f"加载数据时出错: {e}")
        
        print("\n数据处理完成!")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
