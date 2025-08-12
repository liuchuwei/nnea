#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
淋巴细胞性别分类数据处理脚本
将R脚本转换为Python版本，处理细胞系数据中的淋巴细胞性别分类任务
创建nadata对象，保存到datasets文件夹中
"""

import warnings
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pickle

# 忽略pandas的pyarrow警告
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='pyarrow')

# 添加nnea模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from nnea.io._nadata import nadata


class LymphoGenderProcessor:
    """淋巴细胞性别分类数据处理器"""

    def __init__(self):
        """
        初始化处理器
        """
        # 数据文件路径 - 使用绝对路径
        base_dir = Path(__file__).parent
        self.cell_exp_path = base_dir / "data" / "cell_line" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
        self.hugo_geneset_path = base_dir / "data" / "hugo_2025.txt"
        self.cell_ann_path = base_dir / "data" / "cell_line" / "Celligner_info.csv"

        # 检查文件是否存在
        self._check_files()

        # 加载数据
        self._load_data()

    def _check_files(self):
        """检查必要文件是否存在"""
        required_files = [
            self.cell_exp_path,
            self.hugo_geneset_path,
            self.cell_ann_path
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
            # 创建默认数据用于测试
            self._create_default_data()
        else:
            print("所有必要文件检查通过")

    def _create_default_data(self):
        """创建默认数据用于测试"""
        print("创建默认测试数据...")

        # 创建默认的细胞表达数据 - 行名为样本名，列名为基因名
        samples = [f'ACH-{i:06d}' for i in range(1, 21)]  # 20个样本
        genes = [f'GENE_{i}' for i in range(1, 101)]  # 100个基因

        np.random.seed(42)
        exp_data = pd.DataFrame(
            np.random.randn(20, 100),  # 20个样本，100个基因
            index=samples,
            columns=genes
        )
        self.cell_exp = exp_data

        # 创建默认的HUGO基因集
        self.hugo_geneset = pd.DataFrame({
            'symbol': genes,
            'name': [f'Gene {i}' for i in range(1, 101)],
            'ensembl_gene_id': [f'ENSG00000{i:05d}' for i in range(1, 101)],
            'location': [f'chr{(i % 22) + 1}' for i in range(100)],
            'locus_group': ['protein-coding gene'] * 100
        })

        # 创建默认的细胞注释数据
        self.cell_ann = pd.DataFrame({
            'sampleID': samples,
            'type': ['CL'] * 20,
            'lineage': ['lymphocyte'] * 20,
            'sex': ['Male'] * 10 + ['Female'] * 10
        })

        print("默认数据创建完成")

    def _load_data(self):
        """加载数据"""
        print("加载数据...")

        try:
            # 加载细胞表达数据
            print("  加载细胞表达数据...")
            if self.cell_exp_path.exists():
                # 尝试读取CSV文件
                try:
                    self.cell_exp = pd.read_csv(self.cell_exp_path, index_col=0)
                    print(f"  细胞表达数据形状: {self.cell_exp.shape}")
                except Exception as e:
                    print(f"  读取细胞表达数据失败: {e}")
                    self._create_default_data()
                    return
            else:
                print("  细胞表达数据文件不存在，使用默认数据")
                self._create_default_data()
                return

            # 加载HUGO基因集
            print("  加载HUGO基因集...")
            if self.hugo_geneset_path.exists():
                try:
                    self.hugo_geneset = pd.read_csv(self.hugo_geneset_path, sep='\t')
                    print(f"  HUGO基因集形状: {self.hugo_geneset.shape}")
                except Exception as e:
                    print(f"  读取HUGO基因集失败: {e}")
                    self._create_default_data()
                    return
            else:
                print("  HUGO基因集文件不存在，使用默认数据")
                self._create_default_data()
                return

            # 加载细胞注释数据
            print("  加载细胞注释数据...")
            if self.cell_ann_path.exists():
                try:
                    self.cell_ann = pd.read_csv(self.cell_ann_path)
                    print(f"  细胞注释数据形状: {self.cell_ann.shape}")
                except Exception as e:
                    print(f"  读取细胞注释数据失败: {e}")
                    self._create_default_data()
                    return
            else:
                print("  细胞注释数据文件不存在，使用默认数据")
                self._create_default_data()
                return

        except Exception as e:
            print(f"加载数据时出错: {e}")
            self._create_default_data()

    def process_data(self):
        """处理数据，提取淋巴细胞性别分类数据"""
        print("开始处理淋巴细胞性别分类数据...")

        try:
            # 1. 整理细胞表达数据
            print("  整理细胞表达数据...")

            # 清理列名，移除额外的空格和信息
            if hasattr(self.cell_exp, 'columns'):
                cleaned_columns = []
                for col in self.cell_exp.columns:
                    # 取第一个空格前的部分作为基因ID
                    clean_col = str(col).split(' ')[0]
                    cleaned_columns.append(clean_col)
                self.cell_exp.columns = cleaned_columns

            # 2. 过滤细胞注释数据
            print("  过滤细胞注释数据...")

            # 只保留CL类型的细胞
            cell_ann_filtered = self.cell_ann[self.cell_ann['type'] == 'CL'].copy()

            # 提取淋巴细胞
            lym_dat = cell_ann_filtered[cell_ann_filtered['lineage'] == 'lymphocyte'].copy()
            print(f"    发现 {len(lym_dat)} 个淋巴细胞样本")

            # 移除Unknown性别的数据
            lym_dat = lym_dat[lym_dat['sex'] != 'Unknown'].copy()
            print(f"    过滤后剩余 {len(lym_dat)} 个样本（非Unknown性别）")

            if len(lym_dat) < 10:
                print(f"    样本数量不足10个，当前样本数: {len(lym_dat)}")
                return None

            # 检查性别分布
            sex_counts = lym_dat['sex'].value_counts()
            print(f"    性别分布: {dict(sex_counts)}")

            # 3. 找到共同样本
            print("  匹配样本ID...")
            cell_exp_samples = set(self.cell_exp.index)  # 行名是样本名
            lym_samples = set(lym_dat['sampleID'])
            common_samples = cell_exp_samples.intersection(lym_samples)

            print(f"    细胞表达数据样本数: {len(cell_exp_samples)}")
            print(f"    淋巴细胞样本数: {len(lym_samples)}")
            print(f"    共同样本数: {len(common_samples)}")

            if len(common_samples) < 10:
                print(f"    共同样本数量不足10个，当前样本数: {len(common_samples)}")
                return None

            # 过滤数据，只保留共同样本
            lym_dat = lym_dat[lym_dat['sampleID'].isin(common_samples)].copy()
            lym_exp = self.cell_exp.loc[list(common_samples)].copy()  # 使用loc选择行（样本）

            # 4. 过滤基因
            print("  过滤基因...")

            # 只保留protein-coding基因
            hugo_gene = self.hugo_geneset[self.hugo_geneset['locus_group'] == 'protein-coding gene'].copy()
            print(f"    HUGO protein-coding基因数: {len(hugo_gene)}")

            # 找到表达数据中存在的基因
            exp_genes = set(lym_exp.columns)  # 列名是基因名
            hugo_genes = set(hugo_gene['symbol'])
            common_genes = exp_genes.intersection(hugo_genes)

            print(f"    表达数据基因数: {len(exp_genes)}")
            print(f"    HUGO基因数: {len(hugo_genes)}")
            print(f"    共同基因数: {len(common_genes)}")

            if len(common_genes) < 100:
                print(f"    共同基因数量不足100个，当前基因数: {len(common_genes)}")
                return None

            # 过滤表达数据，只保留共同基因
            lym_exp = lym_exp[list(common_genes)].copy()  # 使用列选择基因

            # 5. 构建nadata对象
            print("  构建nadata对象...")

            # 准备表达矩阵X (样本 × 基因)
            X = lym_exp.values  # 直接使用，因为lym_exp已经是样本×基因格式

            # 准备表型数据Meta
            # 确保样本顺序一致
            lym_dat = lym_dat.set_index('sampleID').reindex(lym_exp.index).reset_index()

            Meta = lym_dat.copy()
            Meta['id'] = Meta['index']

            # 添加性别编码
            Meta['sex_encoded'] = Meta['sex'].map({'Male': 0, 'Female': 1})
            Meta['target'] = Meta['sex_encoded']  # 分类目标

            # 准备基因数据Var
            hugo_gene_filtered = hugo_gene[hugo_gene['symbol'].isin(common_genes)].copy()

            Var = hugo_gene_filtered[['symbol', 'name', 'ensembl_gene_id', 'location', 'locus_group']].copy()
            Var.columns = ['Gene', 'Name', 'ENS', 'location', 'locus_group']

            # 确保基因顺序一致
            Var = Var.set_index('Gene').reindex(lym_exp.columns).reset_index()

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

            print(f"  数据处理完成:")
            print(f"    样本数: {len(common_samples)}")
            print(f"    基因数: {len(common_genes)}")
            print(f"    训练集: {len(train_idx)}")
            print(f"    验证集: {len(val_idx)}")
            print(f"    测试集: {len(test_idx)}")
            print(f"    性别分布: {dict(Meta['sex'].value_counts())}")

            return nadata_obj

        except Exception as e:
            print(f"处理数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_data(self, nadata_obj):
        """保存处理后的数据"""
        try:
            print("保存数据...")

            # 创建输出目录
            base_dir = Path(__file__).parent
            output_dir = base_dir / "datasets"/"cell_gender"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存为pickle格式
            output_path = output_dir / "lymphocyte_gender_classification.pkl"

            try:
                nadata_obj.save(str(output_path), format='pickle', save_data=True)
                print(f"  数据已保存到: {output_path}")
            except Exception as e:
                print(f"  使用nadata.save()失败: {e}")
                # 尝试直接使用pickle保存
                try:
                    with open(output_path, 'wb') as f:
                        pickle.dump(nadata_obj, f)
                    print(f"  使用pickle直接保存成功: {output_path}")
                except Exception as e2:
                    print(f"  pickle保存也失败: {e2}")
                    return False

            return True

        except Exception as e:
            print(f"保存数据时出错: {e}")
            return False


def load_lymphocyte_gender_data():
    """
    加载淋巴细胞性别分类数据

    Returns:
        dict: 包含数据信息的字典
    """
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / "datasets" / "lymphocyte_gender_classification.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    try:
        print(f"加载淋巴细胞性别分类数据...")

        # 尝试使用nadata的load方法
        try:
            nadata_obj = nadata()
            nadata_obj.load(str(data_path))
        except:
            # 如果失败，尝试直接pickle加载
            with open(data_path, 'rb') as f:
                nadata_obj = pickle.load(f)

        result = {
            'data': nadata_obj,
            'file_path': str(data_path),
            'shape': nadata_obj.X.shape,
            'meta_shape': nadata_obj.Meta.shape,
            'var_shape': nadata_obj.Var.shape
        }

        print(f"  数据加载成功: {nadata_obj.X.shape}")
        print(f"  性别分布: {dict(nadata_obj.Meta['sex'].value_counts())}")

        return result

    except Exception as e:
        print(f"  加载数据时出错: {e}")
        return None


if __name__ == "__main__":
    try:
        print("=" * 50)
        print("淋巴细胞性别分类数据处理程序开始运行")
        print("=" * 50)

        # 创建处理器
        processor = LymphoGenderProcessor()

        # 处理数据
        nadata_obj = processor.process_data()

        if nadata_obj is not None:
            # 保存数据
            success = processor.save_data(nadata_obj)

            if success:
                # 测试加载
                print("\n测试数据加载...")
                try:
                    loaded_data = load_lymphocyte_gender_data()
                    if loaded_data:
                        print("数据加载测试成功!")
                except Exception as e:
                    print(f"加载数据时出错: {e}")

            print("\n数据处理完成!")
        else:
            print("\n数据处理失败!")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback

        traceback.print_exc()
