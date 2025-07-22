#!/usr/bin/env Python
# coding=utf-8
import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import ConcatDataset

from model.explainer import Explainer
from model.trainer import Trainer
from utils.train_utils import SetSeed, LoadModel, SetDevice
from utils.io_utils import Loader, LoadConfig

import torch

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
    global_config, trainer_config, model_config, data_config = LoadConfig(config, explain=True)
    checkpoints = [f for f in os.listdir(args.model)
                  if os.path.isfile(os.path.join(args.model, f)) and f.lower().endswith('.pt')]
    checkpoints = np.array([int(item.split("_")[0]) for item in checkpoints]).max().astype(str) + "_checkpoint.pt"

    model_config['check_point'] = os.path.join(args.model, checkpoints)
    model_config['indicator'] = os.path.join(args.model,   "_indicator.csv")
    model_config['geneset_importance'] = os.path.join(args.model,  "_gs.csv")

    "2.set seed"
    SetSeed(global_config['seed'])

    "3.set device"
    config = SetDevice(global_config)

    "4.load dataset"
    loader = Loader(global_config, data_config)
    loader.load_dataset()

    "5.build model"
    model = LoadModel( {**global_config, **model_config, **trainer_config}, loader)
    checkpoint = torch.load(model_config['check_point'])
    model.load_state_dict( checkpoint)

    train_loader = {"train": loader.train_dataset, "valid": loader.val_dataset, "targets": loader.targets}
    final_trainer = Trainer({**global_config, **model_config, **trainer_config}, model, train_loader)

    test_loader = torch.utils.data.DataLoader(
        ConcatDataset([loader.test_dataset, loader.train_dataset, loader.val_dataset]),
        batch_size=trainer_config['batch_size'],
        shuffle=False
    )

    final_trainer.evaluate(loader=test_loader)

    print(classification_report(final_trainer.all_targets, final_trainer.all_predictions))
    df = pd.DataFrame({
        'all_targets': final_trainer.all_targets.squeeze(),
        'all_predictions': final_trainer.all_predictions.squeeze(),
        'indice': final_trainer.all_indice.squeeze()
    })
    df.to_csv(os.path.join(args.model, 'evaluate_results.csv'), index=False)

    "6.explain model"
    explainer = Explainer(config, model, loader)
    explainer.explain()

