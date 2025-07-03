import random
import numpy as np
import torch
from typing import *
from torch import optim

from model.nnea_model import nnea


def LoadModel(config):

    if config['model'] == 'nnea':

        model = nnea(config)

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

    return scheduler, optimizer
