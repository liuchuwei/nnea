"""
NNEA (Neural Network with Explainable Architecture) Package

A comprehensive biological interpretable neural network package that integrates model, module, utils, and config functionality,
specifically designed for transcriptomics research. This package provides a complete workflow from data loading, preprocessing, model training to result interpretation.
"""

from . import datasets
from . import io
from . import data_factory
from . import model
from . import plot
from . import utils
from . import config
# from . import baseline_models  # Temporarily commented out because the module doesn't exist
from . import logging_utils

# Main interface functions
from .io import CreateNNEA, nadata
from .model.models import build, train, eval, explain, save_model, load_project, get_summary, train_classification_models, compare_models, print_model_structure, predict
# from .model.classification_models import (  # Temporarily commented out
#     build_classification_models, train_classification_models as train_clf_models, 
#     ClassificationModelComparison
# )
# from .baseline_models import BaselineModelComparison  # Temporarily commented out
# from .plot.simple_plots import training_curve, feature_importance, geneset_network, model_comparison  # Temporarily commented out
# from .cross_validation import (  # Temporarily commented out
#     cross_validation_hyperparameter_search,
#     train_final_model_with_cv,
#     run_cv_experiment,
#     save_cv_results,
#     plot_cv_results,
#     HyperparameterOptimizer,
#     MultiModelCrossValidator,
#     run_multi_model_cv_experiment
# )

# Data preprocessing functionality
from .data_factory import pp
# Feature selection functionality
from .data_factory.feature_selection import fs

# Utility functions
from .utils.helpers import set_global_seed, get_seed_from_config, ensure_reproducibility

# Logging related
from .logging_utils import setup_logging, get_logger, logger

__version__ = "0.1.0"
__author__ = "NNEA Team"
__email__ = "nnea@example.com"

__all__ = [
    "CreateNNEA",
    "nadata",
    "build", 
    "train", 
    "eval", 
    "explain",
    "save_model",
    "load_project", 
    "get_summary",
    "print_model_structure",
    "predict",
    # "build_classification_models",  # Temporarily commented out
    # "train_classification_models",
    # "train_clf_models",
    "compare_models",
    # "ClassificationModelComparison",  # Temporarily commented out
    # "BaselineModelComparison",  # Temporarily commented out
    # "training_curve",  # Temporarily commented out
    # "feature_importance",
    # "geneset_network",
    # "model_comparison",
    # "cross_validation_hyperparameter_search",  # Temporarily commented out
    # "train_final_model_with_cv",
    # "run_cv_experiment",
    # "save_cv_results",
    # "plot_cv_results",
    # "HyperparameterOptimizer",
    # "MultiModelCrossValidator",
    # "run_multi_model_cv_experiment",
    "pp",
    "fs",
    "set_global_seed",
    "get_seed_from_config", 
    "ensure_reproducibility",
    "setup_logging",
    "get_logger",
    "logger"
]