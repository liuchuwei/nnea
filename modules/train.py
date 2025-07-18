#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import torch

from model.trainer import Trainer, CrossTrainer
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

    if global_config['train_mod'] == 'one_split':
        "5.1 build model"
        model = LoadModel(model_config, loader)

        "5.2 train model"
        trainer = Trainer(trainer_config, model, loader)
        trainer.train()

    elif global_config['train_mod'] == 'cross_validation':
        trainer = CrossTrainer(trainer_config, model_config, global_config, loader)
        trainer.train()

