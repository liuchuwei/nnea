#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import torch

from model.trainer import Trainer, CrossTrainer, OneSplitTrainer, ML_Trainer
from utils.train_utils import SetSeed, LoadModel, SetDevice
from utils.io_utils import Loader, LoadConfig


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    parser.add_argument("--config",
                        help='Path to experiment config file.',
                        required=True)
    
    parser.add_argument("--verbose",
                        help='Verbose level: 0=progress bar only, 1=basic info, 2=detailed evaluation',
                        type=int,
                        default=1)

    return parser


def main(args):

    "1.load config"
    global_config, trainer_config, model_config, data_config = LoadConfig(args.config)

    "2.set seed"
    SetSeed(global_config['seed'])

    "3.set device"
    global_config = SetDevice(global_config)

    "4.load dataset"
    loader = Loader(global_config, data_config)
    loader.load_dataset()

    "5.build and train model"

    # 获取verbose参数
    verbose = getattr(args, 'verbose', 1)

    if global_config['model'] in ["LR", "DT", "RF", "AB", "LinearSVM", "RBFSVM", "NN"]:

        trainer = ML_Trainer({**model_config, **global_config, **trainer_config}, loader)
        trainer.train(verbose=verbose)

    elif global_config['model'] in ['nnea']:
        if global_config['train_mod'] == 'one_split':

            trainer = OneSplitTrainer(trainer_config, model_config, global_config, loader)
            trainer.train(verbose=verbose)

        elif global_config['train_mod'] == 'cross_validation':
            trainer = CrossTrainer(trainer_config, model_config, global_config, loader)
            trainer.train(verbose=verbose)

