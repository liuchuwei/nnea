#!/usr/bin/env Python
# coding=utf-8
import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter

from model.explainer import Explainer
from utils.train_utils import SetSeed, LoadModel, SetDevice
from utils.io_utils import Loader, LoadConfig


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    parser.add_argument("--model",
                        help='Path to model directory.',
                        required=True)

    return parser


def main(args):

    "1.load config"
    config = [f for f in os.listdir(args.model)
                  if os.path.isfile(os.path.join(args.model, f)) and f.lower().endswith('.toml')]

    config = os.path.join(args.model, config[0])
    config = LoadConfig(config)
    config['check_point'] = os.path.join(args.model, "_checkpoint.pt")
    config['indicator'] = os.path.join(args.model,   "_indicator.csv")
    config['geneset_importance'] = os.path.join(args.model,  "_gs.csv")

    "2.set seed"
    SetSeed(config['seed'])

    "3.set device"
    config = SetDevice(config)

    "4.load dataset"
    loader = Loader(config)
    loader.load_dataset()

    "5.build model"
    model = LoadModel(config, loader)
    model.build_model()
    model.to(config['device'])

    "6.explain model"
    explainer = Explainer(config, model, loader)
    explainer.explain()

