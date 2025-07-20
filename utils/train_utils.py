import logging
import random
import numpy as np
import torch
from typing import *

from scipy.stats import loguniform, uniform
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import optim, nn
import torch.nn.functional as F

from model.nnea_layers import TrainableGeneSetLayer


def LoadModel(config, loader):

    if config['model'] == 'nnea':
        from model.nnea_model import nnea
        model = nnea(config, loader=loader)
        model.build_model()
        model.to(config['device'])

    elif config['model'] == 'LR':
        print("training logistic regression model...")
        if config['train_mod']=="cross_validation" and config['hyper_C_type'] == 'loguniform':
            model = {
                "model": LogisticRegression(max_iter=config['max_iter'], random_state=config['seed']),
                "params": {
                    "C": loguniform(config['hyper_C_min'], config['hyper_C_max']),
                    "penalty": config['penalty'],
                    "solver": config['solver'],
                    'l1_ratio': uniform(0, 1),  # l1_ratio ∈ [0, 1]
                    "class_weight": config['class_weight']
                }
            }
        elif config['train_mod'] == "one_split":
            model = LogisticRegression(max_iter=config['max_iter'], random_state=config['seed'],
                                       penalty=config['penalty'],solver=config['solver'], C=config['hyper_C'],
                                       class_weight=config['class_weight'])

    elif config['model'] == 'DT':
        print("training decision tree model...")
        if config['train_mod'] == "cross_validation":
            model = {
                "model": DecisionTreeClassifier(random_state=config['seed']),
                "params": {
                    "max_depth": config['max_depth'],
                    "min_samples_split": config['min_samples_split'],
                    "min_samples_leaf": config['min_samples_leaf'],
                    "criterion": config['criterion'],
                    "class_weight": config['class_weight']
                }
            }
        elif config['train_mod'] == "one_split":
            model = DecisionTreeClassifier(
                random_state=config['seed'],
                max_depth=config['max_depth'][0] if isinstance(config['max_depth'], list) else config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                criterion=config['criterion'],
                class_weight=config['class_weight']
            )

    elif config['model'] == 'RF':
        print("training random forest model...")
        if config['train_mod'] == "cross_validation":
            model = {
                "model": RandomForestClassifier(random_state=config['seed']),
                "params": {
                    "n_estimators": config['n_estimators'],
                    "max_depth": config['max_depth'],
                    "min_samples_split": config['min_samples_split'],
                    "max_features": config['max_features'],
                    "class_weight": config['class_weight']
                }
            }
        elif config['train_mod'] == "one_split":
            model = RandomForestClassifier(
                random_state=config['seed'],
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                max_features=config['max_features'],
                class_weight=config['class_weight']
            )
    elif config['model'] == 'AB':
        print("training adaptive boosting model...")
        if config['train_mod'] == "cross_validation":
            model = {
                "model": AdaBoostClassifier(random_state=config['seed']),
                "params": {
                    "n_estimators": config['n_estimators'],
                    "learning_rate": config['learning_rate'],
                    "algorithm": config['algorithm']
                }
            }
        elif config['train_mod'] == "one_split":
            model = AdaBoostClassifier(
                random_state=config['seed'],
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                algorithm=config['algorithm']
            )

    elif config['model'] in ['LinearSVM', 'RBFSVM']:
        kernel_type = 'linear' if config['model'] == 'LinearSVM' else 'rbf'
        print(f"training {kernel_type} SVM model...")

        if config['train_mod'] == "cross_validation":
            params = {
                "C": loguniform(config['hyper_C_min'], config['hyper_C_max']),
                "kernel": [kernel_type],
                "class_weight": config['class_weight']
            }
            if kernel_type == 'rbf':
                params["gamma"] = loguniform(config['gamma_min'], config['gamma_max'])

            model = {
                "model": SVC(max_iter=config['max_iter'], random_state=config['seed']),
                "params": params
            }
        elif config['train_mod'] == "one_split":
            params = {
                "C": config['hyper_C'],
                "kernel": kernel_type,
                "class_weight": config['class_weight']
            }
            if kernel_type == 'rbf':
                params["gamma"] = config['gamma']
            model = SVC(max_iter=config['max_iter'], random_state=config['seed'], **params)

    elif config['model'] == 'NN':
        print("training neural network model...")
        if config['train_mod'] == "cross_validation":
            model = {
                "model": MLPClassifier(random_state=config['seed'], max_iter=config['max_iter']),
                "params": {
                    "hidden_layer_sizes": config['hidden_layer_sizes'],
                    "activation": config['activation'],
                    "solver": config['solver'],
                    "alpha": loguniform(config['alpha_min'], config['alpha_max']),
                    "learning_rate": config['learning_rate']
                }
            }
        elif config['train_mod'] == "one_split":
            model = MLPClassifier(
                random_state=config['seed'],
                hidden_layer_sizes=config['hidden_layer_sizes'],
                activation=config['activation'],
                solver=config['solver'],
                alpha=config['alpha'],
                learning_rate=config['learning_rate'],
                max_iter=config['max_iter']
            )

    return model

def SetSeed(seed: Optional[int] = 1):

    """
    Method to set global training seed for repeatability of experiment

    :param seed: seed number
    :return: none
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def SetDevice(config):

    """
    Setting device
    :param config:
    :return: config
    """

    if config['device'] == "cuda" and torch.cuda.is_available():
        config['device'] = "cuda"
    else:
        config['device'] = "cpu"
    print("Setting device: {}".format(config['device']))

    return config


def BuildDeepGenesetLayer(config, prior_tensor = None):
    """
    Method to build deep geneset layer

    :param config: 
    :return: gene_set_layer nn.ModuleList()
    """

    gene_set_layers = nn.ModuleList()

    for i in range(config['geneset_layers']):
        config['geneset_dropout'] = config['deep_dropout'][i]
        if i == 0 and prior_tensor is not None:
            layer = TrainableGeneSetLayer(
                num_genes=config['num_genes'],
                num_sets=prior_tensor.shape[0],
                min_set_size=config['set_min_size'],
                max_set_size=config['set_max_size'],
                alpha=config['alpha'],
                is_deep_layer=True,
                layer_index=i,
                prior_knowledge=prior_tensor,
                freeze_prior=config['freeze_prior'],
                use_attention=config['use_attention'],
                attention_dim=config['attention_dim']
            )
        else:
            layer = BuildGenesetLayer(config, prior_tensor)
        gene_set_layers.append(layer)
    
    return gene_set_layers

        
def BuildGenesetLayer(config, prior_tensor = None):

    """
    Method to build geneset layer
    :param config:
    :return: gene_set_layer
    """
    
    gene_set_layer = TrainableGeneSetLayer(
        num_genes=config['num_genes'],
        num_sets=config['num_sets'],
        min_set_size=config['set_min_size'],
        max_set_size=config['set_max_size'],
        alpha=config['geneset_layer_alpha'],
        is_deep_layer=False,
        prior_knowledge=prior_tensor,
        freeze_prior=config['freeze_prior'],
        num_fc_layers=config['num_fc_layers'],
        geneset_dropout=config['geneset_dropout'],
        use_attention=config['use_attention'],
        attention_dim=config['attention_dim']
    )
        

    return gene_set_layer


class AttentionBlock(nn.Module):
    """自注意力模块，用于动态加权特征"""

    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)  # 压缩通道降低计算量
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的残差权重

    def forward(self, x):
        # x shape: [Batch, Sequence, Features] → 需调整为特征维度
        if x.dim() == 2:  # 若输入为2D（无序列长度），增加序列维度
            x = x.unsqueeze(1)  # [B, 1, D]

        Q = self.query(x)  # [B, Seq, D//8]
        K = self.key(x)  # [B, Seq, D//8]
        V = self.value(x)  # [B, Seq, D]

        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2))  # [B, Seq, Seq]
        attn_weights = F.softmax(scores, dim=-1)  # 归一化权重

        # 加权聚合特征
        context = torch.bmm(attn_weights, V)  # [B, Seq, D]

        # 残差连接 + 特征压缩回原始维度
        output = self.gamma * context + x
        return output.squeeze(1)  # 移除序列维度 → [B, D]
def BuildClassifier(config, input_dim):

    if config['classifier_name'] == "linear":

        layers = []
        current_dim = input_dim
        hidden_dims = config['hidden_dims']

        if len(hidden_dims)>0:
            for i, h_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                if i != len(hidden_dims) - 1:
                    layers.append(nn.BatchNorm1d(h_dim))
                    layers.append(nn.Dropout(config['classifier_dropout'][i]))
                current_dim = h_dim
        else:
            # 输出层
            layers.append(nn.ReLU())
        layers.append(nn.Linear(current_dim, config['output_dim']))

        classifier = nn.Sequential(*layers)

    elif config['classifier_name'] == "attention":
        layers = []
        current_dim = input_dim

        # 1. 添加自注意力层
        layers.append(AttentionBlock(current_dim))

        # 2. 添加MLP层（与linear分类器结构一致）
        hidden_dims = config['hidden_dims']
        if len(hidden_dims) > 0:
            for i, h_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                if i != len(hidden_dims) - 1:
                    layers.append(nn.BatchNorm1d(h_dim))
                    layers.append(nn.Dropout(config['classifier_dropout'][i]))
                current_dim = h_dim

        # 3. 输出层
        layers.append(nn.Linear(current_dim, config['output_dim']))
        classifier = nn.Sequential(*layers)

    return classifier

def cox_loss(risks, times, events):
    # 对生存时间排序
    sort_idx = torch.argsort(times, descending=True)
    risks = risks[sort_idx]
    events = events[sort_idx]

    # 计算损失
    loss = 0
    for i in range(len(events)):
        if events[i] == 1:
            # 计算当前事件的风险集
            risk_set = risks[i:]
            # 使用log-sum-exp技巧避免数值溢出
            max_risk = torch.max(risk_set)
            log_sum_exp = max_risk + torch.log(torch.sum(torch.exp(risk_set - max_risk)))
            # 累加每个事件贡献的损失
            loss += (log_sum_exp - risks[i])

    return loss / torch.sum(events)  # 归一化



def BuildOptimizer(params, config=None):

    r"""
    instance method for building optimizer

        Args:
            params: model params
            config: optimizer config

        Return:
            none

    """
    filter_fn = filter(lambda p: p.requires_grad, params)

    if config['opt'] == 'adam':
        optimizer = optim.Adam(filter_fn, lr=config['lr'],
                               weight_decay=config['weight_decay'])
    elif config['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(filter_fn, lr=config['lr'],
                                      weight_decay=config['weight_decay'],
                                      amsgrad=config['amsgrad'])
    elif config['opt'] == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=config['lr'],
                              # momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    elif config['opt'] == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    elif config['opt'] == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    if config['opt_scheduler'] == 'none':
        return None, optimizer
    elif config['opt_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['opt_decay_step'],
                                              gamma=config['opt_decay_rate'])
    elif config['opt_scheduler'] == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['opt_restart'])

    elif config['opt_scheduler'] == 'reduce':

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['opt_factor'], patience=config['opt_patience'])


    return scheduler, optimizer
