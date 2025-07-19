import logging
import random
import numpy as np
import torch
from typing import *

from scipy.stats import loguniform, uniform
from sklearn.linear_model import LinearRegression, LogisticRegression
from torch import optim, nn

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
                freeze_prior=config['freeze_prior']
            )
        else:
            layer = TrainableGeneSetLayer(
                num_genes=config['num_genes'],
                num_sets=prior_tensor.shape[0],
                min_set_size=config['set_min_size'],
                max_set_size=config['set_max_size'],
                alpha=config['alpha'],
                is_deep_layer=True,
                layer_index=i,
                prior_knowledge=None,
                freeze_prior=config['freeze_prior'],
                num_fc_layers=config['num_fc_layers'],
            )
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
        geneset_dropout=config['geneset_dropout']
    )
        

    return gene_set_layer
    

def BuildClassifier(config, input_dim):

    if config['classifier_name'] == "linear":

        layers = []
        current_dim = input_dim
        hidden_dims = config['hidden_dims']

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            if i != len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.Dropout(config['classifier_dropout'][i]))
            current_dim = h_dim

        # 输出层
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
                              momentum=config['momentum'],
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
