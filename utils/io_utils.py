import os

import h5py
import numpy as np
import pandas as pd
import torch
import toml
import time

from scipy.sparse import csc_matrix, csr_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler


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

    # split config
    global_config = config['global']
    train_config = config['trainer']
    data_config = config['dataload']


    # extract config
    if global_config['model'] == "LR":
        model_config = config["LR"]
    elif global_config['model'] == "nnea":
        model_config = flatten_dict(config["nnea"])

    "flatten config dictionary"
    # config = flatten_dict(config)
    global_config['toml_path'] = path

    "generate storage path"
    formatted_date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    checkpoint_dir = "checkpoints/" + formatted_date + "_" + data_config['dataset'] + "_" + global_config['model']
    model_config['checkpoint_dir'] = checkpoint_dir

    "define task"
    if global_config['task'] in ["cell_drug", "cell_dependency", "regression"]:
        global_config['task'] = "regression"
    elif global_config['task'] in ["tumor_immunotherapy", "cell_class", "tumor_drug", "sc_classification", "sc_annotation", "classification"] :
        global_config['task'] = "classification"
    elif global_config['task'] in ["tumor_sur", "cox"]:
        global_config['task'] = "cox"
    elif global_config['task'] in ["sc_dimension_reduction", "autoencoder"]:
        global_config['task'] = "autoencoder"
    elif global_config['task'] == "sc_umap":
        global_config['task'] = "umap"

    return global_config, train_config, model_config, data_config
class Loader(object):

    def __init__(self, global_config = None, config=None):

        self.global_config = global_config
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

        if self.global_config['model'] in ['nnea']:
            self.load_torch_dataset(rank_exp, sort_exp, norm_exp, phe, pca)

        elif self.global_config['model'] in ['LR']:

            X = norm_exp
            if self.config['scaler'] == "mean_sd":
                X = scale_dense_matrix(X, standardize_method="mean_sd")

            elif self.config['scaler'] == "min_max":
                X = scale_dense_matrix(X, standardize_method="min_max")

            if self.config['top_gene']:
                X = retain_top_var_gene(X, top_gene=self.config['top_gene'])

            y = phe.flatten()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], random_state=self.global_config['seed']
            )

            if self.config['scaler'] == "mean_sd":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if self.config['strategy'] == "StratifiedKFold":
                cv = StratifiedKFold(
                    n_splits=self.config['n_splits'],
                    shuffle=self.config['shuffle'],
                    random_state=self.config['seed']
                )
            elif self.config['strategy'] == "KFold":
                cv = KFold(
                    n_splits=self.config['n_splits'],
                    shuffle=self.config['shuffle'],
                    random_state=self.config['seed']
                )

            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.cv = cv

    def load_torch_dataset(self, rank_exp, sort_exp, norm_exp, phe, pca):

        rank_exp_tensor = torch.tensor(rank_exp, dtype=torch.float32)
        sort_exp_tensor = torch.tensor(sort_exp, dtype=torch.long)

        if self.global_config['task'] == "cox":

            "generate torch dataset"
            # 索引需用long类型
            events_tensor = torch.tensor(phe[:,0], dtype=torch.float32)
            times_tensor = torch.tensor(phe[:,1], dtype=torch.float32)

            base_dataset = (rank_exp_tensor, sort_exp_tensor, times_tensor, events_tensor)
            targets = phe[:, 0]


        elif self.global_config['task'] in ['classification', "regression"]:

            y_tensor = torch.tensor(phe, dtype=torch.float32)
            base_dataset = (rank_exp_tensor, sort_exp_tensor, y_tensor)
            targets = phe.flatten()  # 目标变量作为分层依据

        elif self.global_config['task'] in ['autoencoder']:

            norm_exp_tensor = torch.tensor(norm_exp, dtype=torch.float32)
            base_dataset = (rank_exp_tensor, sort_exp_tensor, norm_exp_tensor)
            targets = None

        elif self.global_config['task'] in ['umap']:

            pca_tensor = torch.tensor(pca[:,:self.config['pca_dim']], dtype=torch.float32)
            y_tensor = torch.tensor(phe, dtype=torch.float32)
            base_dataset = (rank_exp_tensor, sort_exp_tensor, pca_tensor, y_tensor)
            targets = phe.flatten()

        self.cv_loaders = []
        self.targets = targets
        self.split_dataset(base_dataset, targets)

    def split_dataset(self, base_dataset, targets):

        n_samples = base_dataset[0].shape[0]
        indices = np.arange(n_samples)

        stratify = targets if self.global_config['task'] == 'classification' else None

        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.config['test_size'],
            stratify=stratify,  # 添加分层抽样
            random_state=self.global_config['seed']
        )

        # 创建训练集和测试集的TensorDataset
        train_dataset = torch.utils.data.TensorDataset(
            *[tensor[train_idx] for tensor in base_dataset]
        )
        test_dataset = torch.utils.data.TensorDataset(
            *[tensor[test_idx] for tensor in base_dataset]
        )

        # 创建测试集loader（不洗牌）
        # self.test_loader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=self.config['batch_size'],
        #     shuffle=False
        # )
        self.test_dataset = test_dataset
        # 保存测试集索引
        self.test_indices = test_idx

        # 生成交叉验证划分
        if self.global_config['train_mod'] == "cross_validation":
            self.split_cross_validation(train_dataset, targets[train_idx])

        elif self.global_config['train_mod'] == "one_split":

            stratify = targets[train_idx] if self.global_config['task'] == 'classification' else None

            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=self.config['val_size'],
                stratify=stratify,  # 添加分层抽样
                random_state=self.global_config['seed']
            )
            self.train_dataset = torch.utils.data.TensorDataset(
                *[tensor[train_idx] for tensor in base_dataset]
            )
            self.val_dataset = torch.utils.data.TensorDataset(
                *[tensor[val_idx] for tensor in base_dataset]
            )

    def split_cross_validation(self, base_dataset, targets=None):
        """生成交叉验证数据划分"""
        n_samples =  len(base_dataset)
        indices = np.arange(n_samples)

        # 选择交叉验证策略
        if self.config['strategy'] == "StratifiedKFold":
            cv = StratifiedKFold(
                n_splits=self.config['n_splits'],
                shuffle=self.config['shuffle'],
                random_state=self.global_config['seed']
            )
            splits = cv.split(indices, targets)
        elif self.config['strategy'] == "KFold":
            cv = KFold(
                n_splits=self.config['n_splits'],
                shuffle=self.config['shuffle'],
                random_state=self.global_config['seed']
            )
            splits = cv.split(indices)

        tensors = base_dataset.tensors

        # 创建每个fold的数据加载器
        for fold, (train_idx, valid_idx) in enumerate(splits):
            # 创建训练集和验证集
            train_dataset = torch.utils.data.TensorDataset(
                *[tensor[train_idx] for tensor in tensors]
            )
            valid_dataset = torch.utils.data.TensorDataset(
                *[tensor[valid_idx] for tensor in tensors]
            )

            # 创建数据加载器
            # train_loader = torch.utils.data.DataLoader(
            #     train_dataset,
            #     batch_size=self.config['batch_size'],
            #     shuffle=True
            # )
            # valid_loader = torch.utils.data.DataLoader(
            #     valid_dataset,
            #     batch_size=self.config['batch_size'],
            #     shuffle=False
            # )

            self.cv_loaders.append({
                'fold': fold,
                'train': train_dataset,
                'valid': valid_dataset,
                'train_indices': train_idx,
                'valid_indices': valid_idx
            })

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
            # norm_exp = normalize_matrix_dense_cell_by_gene(mat=(exp.drop(exp.columns[0], axis=1)))
            norm_exp = exp.drop(exp.columns[0], axis=1).T

            dataset = {
                "gene": gene,
                "sample_id": sample,
                "rank_exp": rank_exp,
                "sort_exp": sort_exp,
                "norm_exp": norm_exp,
                "pca": np.array([0]),
                "phe": phe
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


import numpy as np


def normalize_matrix_dense_cell_by_gene(mat, scale_factor=10000):
    """
    细胞×基因密集矩阵归一化（行为细胞，列为基因）

    参数:
    mat : numpy.ndarray
        密集矩阵，行对应细胞，列对应基因
    scale_factor : int, 默认10000
        缩放因子
    standardize_method : str, 可选
        标准化方法: 'min_max', 'mean_sd' 或 None(默认)

    返回:
    numpy.ndarray - 处理后的矩阵
    """
    # 转换输入为numpy数组
    mat = np.asarray(mat)
    total_umi = np.sum(mat, axis=1, keepdims=True)

    # 避免除零错误
    total_umi[total_umi == 0] = 1e-12

    # 执行CPM归一化和log变换
    scaled_matrix = (mat / total_umi) * scale_factor
    normalized_matrix = np.log1p(scaled_matrix)

    return normalized_matrix


def scale_dense_matrix(normalized_matrix, standardize_method="mean_sd"):

    # 执行指定标准化方法
    if standardize_method == 'min_max':
        # 按基因(列)进行min-max标准化
        col_min = np.min(normalized_matrix, axis=0, keepdims=True)
        col_max = np.max(normalized_matrix, axis=0, keepdims=True)

        # 避免分母为零
        range_mask = (col_max - col_min) > 1e-12
        standardized = np.empty_like(normalized_matrix)

        # 仅对变化范围>1e-12的基因标准化
        standardized[:, range_mask.flatten()] = (
                (normalized_matrix[:, range_mask.flatten()] - col_min[:, range_mask.flatten()]) /
                (col_max[:, range_mask.flatten()] - col_min[:, range_mask.flatten()])
        )
        # 保持常量值不变
        standardized[:, ~range_mask.flatten()] = 0.0

        return standardized

    elif standardize_method == 'mean_sd':
        # 按基因(列)进行z-score标准化
        col_mean = np.mean(normalized_matrix, axis=0, keepdims=True)
        col_std = np.std(normalized_matrix, axis=0, keepdims=True)

        # 避免除零错误
        std_mask = col_std > 1e-12
        standardized = np.empty_like(normalized_matrix)

        # 仅对标准差>1e-12的基因标准化
        standardized[:, std_mask.flatten()] = (
                (normalized_matrix[:, std_mask.flatten()] - col_mean[:, std_mask.flatten()]) /
                col_std[:, std_mask.flatten()]
        )
        # 保持常量值不变
        standardized[:, ~std_mask.flatten()] = 0.0

        return standardized


def retain_top_var_gene(exp, top_gene):
    variances = np.var(exp, axis=0)  # 形状 (18065,)
    top_2000_idx = np.argsort(variances)[::-1][:top_gene]  # 降序排列取前2000
    exp = exp[:, top_2000_idx]

    return exp
