#!/usr/bin/env Python
# coding=utf-8
import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter

import numpy as np
import toml
from utils.io_utils import Loader


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
    config = toml.load(args.config)

    "2.generate dataset"
    loader = Loader(config=config)
    loader.generate_dataset()
