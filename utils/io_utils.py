import os

import numpy as np
import pandas as pd
import torch
import toml
import time


def flatten_dict(nested_dict, parent_key='', sep='.'):
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

    config = toml.load(path)
    # params = "_".join([f"{key}_{value}" for key, value in config['params'].items()])
    config = flatten_dict(config)
    config['path'] = path
    formatted_date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    checkpoint_dir = "checkpoints/" + formatted_date + "_" + config['dataset']
    config['checkpoint_dir'] = checkpoint_dir
    config['check_point'] = os.path.join(config['checkpoint_dir'],   "_checkpoint.pt")
    config['indicator'] = os.path.join(config['checkpoint_dir'],   "_indicator.csv")
    config['geneset_importance'] = os.path.join(config['checkpoint_dir'],  "_gs.csv")

    return config
class Loader(object):

    def __init__(self, config=None):

        self.config = config

    def expToRank(self, mat):
        rank_exp = mat.rank(ascending=True, axis=1)
        sort_exp = np.argsort(rank_exp, axis=1)
        sort_exp = sort_exp[:, ::-1]

        return rank_exp, sort_exp

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

    def load(self):

        if self.config['dataset'] == "lym_sex":

            print("load lym_sex dataset...")
            exp = pd.read_csv('data/lym_exp.csv')
            sex = pd.read_csv('data/lym_sex.csv')

            gender_mapping = {'Male': 1, 'Female': 0}
            sex['gender'] = sex['x'].map(gender_mapping)

            self.num_genes = exp.shape[1]-1
            self.num_class = 2
            self.genes_name = exp.columns[1:].tolist()
            # exp to rank
            rank_exp, sort_exp = self.expToRank(exp.drop(exp.columns[0], axis=1))

            # transform to torch tensor
            rank_exp_tensor = torch.tensor(rank_exp.values, dtype=torch.float32)
            sort_exp_tensor = torch.tensor(sort_exp.copy(), dtype=torch.long)  # 索引需用long类型
            y_tensor = torch.tensor(sex['gender'].values, dtype=torch.long)

            self.torchLoader(rank_exp_tensor, sort_exp_tensor, y_tensor)

def Normalize_matrix(mat, scale_factor=10000):
    """
    对基因表达矩阵进行归一化处理（模拟Seurat/Scanpy的归一化流程）

    参数:
    mat : numpy.ndarray
        基因表达矩阵，行为基因，列为细胞
    scale_factor : int, 默认10000
        缩放因子（TPM归一化基数）

    返回:
    numpy.ndarray - 归一化后的矩阵
    """
    # 1. 计算每个细胞的总UMI计数（列方向求和）
    total_umi = np.sum(mat.values, axis=1, keepdims=True)

    # 处理全零列：避免除以0错误
    total_umi[total_umi == 0] = 1e-12  # 将0替换为极小值

    # 2. 归一化 + 缩放
    # 公式: (原始值 / 细胞总UMI) * scale_factor
    scaled_matrix = (mat / total_umi) * scale_factor

    # 3. log1p转换（自然对数）
    # 相当于 log(scaled_matrix + 1)
    normalized_matrix = np.log1p(scaled_matrix)

    return normalized_matrix