#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import torch

from model.trainer import Trainer
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
    config = LoadConfig(args.config)

    "2.set seed"
    SetSeed(config['seed'])

    "3.set device"
    config = SetDevice(config)

    "4.load dataset"
    loader = Loader(config)
    loader.load_dataset()

    "5.build model"
    model = LoadModel(config, loader)

    "5.train model"
    trainer = Trainer(config, model, loader)
    trainer.train()

