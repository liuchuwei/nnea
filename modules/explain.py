#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import toml
import torch

from model.explainer import Explainer
from utils.train_utils import SetSeed, LoadModel
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

    "3.load dataset"
    loader = Loader(config)
    loader.load()
    config['num_classes'] = loader.num_class
    config['num_genes'] = loader.num_genes

    "4.build model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LoadModel(config)
    model.build_model()
    model.to(device)

    "5.explain model"
    explainer = Explainer(config, model, loader)
    explainer.explain()

