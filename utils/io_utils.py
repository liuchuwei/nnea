import os

import h5py
import numpy as np
import pandas as pd
import torch
import toml
import time

from scipy.sparse import csc_matrix, csr_matrix


def flatten_dict(nested_dict, parent_key='', sep='.'):

    """
    Flattens a nested dictionary into a flat dictionary.
    :param nested_dict:
    :param parent_key:
    :param sep:
    :return: flat dictionary.
    """
    flattened = {}
    for key, value in nested_dict.items():
        # new_key = f"{parent_key}{sep}{key}" if parent_key else key
        new_key = key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep))
        else:
            flattened[new_key] = value
    return flattened


def LoadConfig(path):

    """
    load config from toml file

    :param path: toml file path
    :return: config dictionary
    """

    config = toml.load(path)

    "flatten config dictionary"
    config = flatten_dict(config)
    config['toml_path'] = path

    "generate storage path"
    formatted_date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    checkpoint_dir = "checkpoints/" + formatted_date + "_" + config['dataset']
    config['checkpoint_dir'] = checkpoint_dir
    config['check_point'] = os.path.join(config['checkpoint_dir'],   "_checkpoint.pt")

    "define task"
    if config['task'] in ["cell_drug", "cell_dependency", "regression"]:
        config['task'] = "regression"
    elif config['task'] in ["cell_class", "tumor_drug", "sc_classification", "sc_annotation", "classification"] :
        config['task'] = "classification"
    elif config['task'] in ["tumor_sur", "cox"]:
        config['task'] = "cox"
    elif config['task'] in ["sc_dimension_reduction", "autoencoder"]:
        config['task'] = "autoencoder"
    elif config['task'] == "sc_umap":
        config['task'] = "umap"

    return config
class Loader(object):

    def __init__(self, config=None):

        self.config = config

    def expToRank(self, mat):
        rank_exp = mat.rank(ascending=True, axis=1)
        sort_exp = np.argsort(rank_exp, axis=1)
        sort_exp = sort_exp[:, ::-1]

        return rank_exp, sort_exp

    from scipy.sparse import csr_matrix
    def expToRank_sparse(self, mat):
        csr_mat = mat.tocsr()
        n_rows, n_cols = csr_mat.shape
        rank_exp = np.zeros((n_rows, n_cols), dtype=np.float32)  # 二维矩阵
        sort_exp = np.zeros((n_rows, n_cols), dtype=np.int32)

        for i in range(n_rows):
            row_start = csr_mat.indptr[i]
            row_end = csr_mat.indptr[i + 1]
            col_indices = csr_mat.indices[row_start:row_end]
            row_data = csr_mat.data[row_start:row_end]

            if len(row_data) == 0:
                continue  # 跳过空行

            # 非零值排名（升序转降序）
            ranks = np.argsort(row_data).argsort() + 1
            rank_exp[i, col_indices] = ranks  # 仅填充非零位置

            # 生成降序排序索引
            sorted_idx = col_indices[np.argsort(-row_data)]
            sort_exp[i, :len(sorted_idx)] = sorted_idx  # 填充有效索引

        return rank_exp, sort_exp  # 维度均为 (n_rows, n_cols)

    def torchLoader(self, rank_exp_tensor, sort_exp_tensor, y_tensor):

        dataset = torch.utils.data.TensorDataset(
            rank_exp_tensor,
            sort_exp_tensor,
            y_tensor
        )
        self.torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

    def load_dataset(self):

        print("Loading dataset: %s..." % self.config['dataset'])
        dataset_path = os.path.join('datasets', self.config['dataset'] + ".h5")

        with h5py.File(dataset_path, 'r') as hf:
            self.gene = [x.decode('utf-8') for x in hf['gene'][:]]  # 处理字符串编码
            rank_exp = hf['rank_exp'][:]
            sort_exp = hf['sort_exp'][:]
            norm_exp = hf['norm_exp'][:]
            pca = hf['pca'][:]
            phe = hf['phe'][:]
            self.sample_ids = [x.decode('utf-8') for x in hf['sample_id'][:]]

        rank_exp_tensor = torch.tensor(rank_exp, dtype=torch.float32)
        sort_exp_tensor = torch.tensor(sort_exp, dtype=torch.long)

        if self.config['task'] == "cox":

            "generate torch dataset"
            # 索引需用long类型
            events_tensor = torch.tensor(phe[:,0], dtype=torch.float32)
            times_tensor = torch.tensor(phe[:,1], dtype=torch.float32)


            dataset = torch.utils.data.TensorDataset(
                rank_exp_tensor,
                        sort_exp_tensor,
                        times_tensor,
                        events_tensor
            )

        elif self.config['task'] in ['classification', "regression"]:

            y_tensor = torch.tensor(phe, dtype=torch.float32)

            dataset = torch.utils.data.TensorDataset(
                rank_exp_tensor,
                sort_exp_tensor,
                y_tensor
            )

        elif self.config['task'] in ['autoencoder']:

            norm_exp_tensor = torch.tensor(norm_exp, dtype=torch.float32)

            dataset = torch.utils.data.TensorDataset(
                rank_exp_tensor,
                sort_exp_tensor,
                norm_exp_tensor
            )
        elif self.config['task'] in ['umap']:

            pca_tensor = torch.tensor(pca[:,:self.config['pca_dim']], dtype=torch.float32)
            y_tensor = torch.tensor(phe, dtype=torch.float32)

            dataset = torch.utils.data.TensorDataset(
                rank_exp_tensor,
                sort_exp_tensor,
                pca_tensor,
                y_tensor
            )

        self.torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

    def storage_dataset(self, dataset):


        dataset_path = os.path.join('datasets', self.config['name'] + ".h5")

        with h5py.File(dataset_path, 'w') as hf:

            gene, sample, rank_exp, sort_exp, norm_exp, pca, phe = dataset['gene'], dataset['sample_id'], dataset['rank_exp'], dataset['sort_exp'], dataset['norm_exp'], dataset['pca'], dataset['phe']

            # 存储基因名称（变长字符串）
            dt = h5py.special_dtype(vlen=str)
            gene_dset = hf.create_dataset("gene", (len(gene),), dtype=dt)
            gene_dset[:] = gene

            # 存储排名矩阵（转换为float32节省空间）
            hf.create_dataset("rank_exp", data=rank_exp.astype(np.float32))

            # 存储排序索引（转换为int32节省空间）
            hf.create_dataset("sort_exp", data=sort_exp.astype(np.int32))

            # 存储排序索引（转换为int32节省空间）
            hf.create_dataset("norm_exp", data=norm_exp.astype(np.float32))

            hf.create_dataset("pca", data=pca.astype(np.float32))

            # 存储生存矩阵（转换为float32节省空间）
            hf.create_dataset("phe", data=phe.astype(np.float32))

            # 存储样本ID（可选但推荐）
            sample_dset = hf.create_dataset("sample_id", (len(sample),), dtype=dt)
            sample_dset[:] = sample

    def generate_dataset(self):

        print("Generating dataset: %s" % self.config['name'])

        if self.config['task'] == "single_cell":

            path = os.path.join('factory', self.config['task'], "sc.h5")

            with h5py.File(path, 'r') as f:
                i = f['i'][:].astype(int) - 1  # 转回0-based索引[6](@ref)
                p = f['p'][:]
                x = f['x'][:]
                dims = tuple(f['dims'][:])
                phe = f['phe'][:]
                rownames = f['rownames'][:] if 'rownames' in f else None
                colnames = f['colnames'][:] if 'colnames' in f else None
                pca = f['pca'][:]
                pca = np.transpose(pca)
                # 重建dgCMatrix（Python中为csc_matrix）
                mat = csc_matrix((x, i, p), shape=dims).T.tocsr()
                if rownames is not None:
                    mat.rows = rownames  # 自定义属性存储行名
                if colnames is not None:
                    mat.cols = colnames  # 自定义属性存储列名

                rank_exp, sort_exp = self.expToRank_sparse(mat)
                gene = rownames.astype(str)
                sample = colnames.astype(str)
                norm_exp = normalize_matrix_sparse_cell_by_gene(mat=mat)

            dataset = {
                "gene": gene,
                "sample_id": sample,
                "rank_exp": rank_exp,
                "sort_exp": sort_exp,
                "norm_exp": norm_exp.toarray(),
                "pca": pca,
                "phe": phe
            }

        else:
            path = os.path.join('factory', self.config['task'])
            exp_path = os.path.join(path, 'exp.txt')
            phe_path = os.path.join(path, 'phe.txt')

            exp = pd.read_csv(exp_path)
            phe = pd.read_csv(phe_path)

            gene = exp["Gene"].tolist()
            rank_exp, sort_exp = self.expToRank(exp.drop(exp.columns[0], axis=1).T)
            sample = rank_exp.index.tolist()
            norm_exp = normalize_matrix_dense_cell_by_gene(mat=exp)

            dataset = {
                "gene": gene,
                "sample_id": sample,
                "rank_exp": rank_exp.values(),
                "sort_exp": sort_exp,
                "norm_exp": norm_exp,
                "pca": None,
                "phe": phe.values()
            }

        self.storage_dataset(dataset)

        print("finish!")


import numpy as np
from scipy.sparse import issparse, csr_matrix, csc_matrix


def normalize_matrix_sparse_cell_by_gene(mat, scale_factor=10000):
    """
    细胞×基因稀疏矩阵归一化（行为细胞，列为基因）

    参数:
    mat : scipy.sparse.csr_matrix
        稀疏矩阵，行对应细胞，列对应基因
    scale_factor : int, 默认10000
        缩放因子

    返回:
    scipy.sparse.csr_matrix - 归一化后的稀疏矩阵
    """
    if not issparse(mat):
        mat = csr_matrix(mat)
    elif not isinstance(mat, csr_matrix):
        mat = mat.tocsr()

    # 计算每个细胞的总UMI（行求和）
    total_umi = np.array(mat.sum(axis=1)).flatten()

    # 避免除零错误
    zero_mask = (total_umi == 0)
    total_umi[zero_mask] = 1e-12

    # 构造对角缩放矩阵
    scaling_factors = scale_factor / total_umi
    diag_scaler = csr_matrix((scaling_factors,
                              (range(len(scaling_factors)),
                               range(len(scaling_factors)))))

    # 执行行归一化
    scaled_matrix = diag_scaler.dot(mat)
    scaled_matrix.data = np.log1p(scaled_matrix.data)

    return scaled_matrix


# 新增归一化函数（密集矩阵版）
def normalize_matrix_dense_cell_by_gene(mat, scale_factor=10000):
    """
    细胞×基因密集矩阵归一化（行为细胞，列为基因）

    参数:
    mat : numpy.ndarray
        密集矩阵，行对应细胞，列对应基因
    scale_factor : int, 默认10000
        缩放因子

    返回:
    numpy.ndarray - 归一化后的矩阵
    """
    # 计算每个细胞的总UMI（行求和）
    total_umi = np.sum(mat, axis=1, keepdims=True)

    # 避免除零错误
    total_umi[total_umi == 0] = 1e-12

    # 执行归一化
    scaled_matrix = (mat / total_umi) * scale_factor
    normalized_matrix = np.log1p(scaled_matrix)

    return normalized_matrix