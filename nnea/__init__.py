"""
NNEA (Neural Network with Explainable Architecture) 包

一个集成了model、module、utils、config功能的生物学可解释性神经网络包，
专门用于转录组学研究。该包提供了从数据加载、预处理、模型训练到结果解释的完整流程。
"""

from . import datasets
from . import io
from . import data_factory
from . import model
from . import plot
from . import utils
from . import config
# from . import baseline_models  # 暂时注释掉，因为模块不存在
from . import logging_utils

# 主要接口函数
from .io import CreateNNEA, nadata
from .model.models import build, train, eval, explain, save_model, load_project, get_summary, train_classification_models, compare_models
# from .model.classification_models import (  # 暂时注释掉
#     build_classification_models, train_classification_models as train_clf_models, 
#     ClassificationModelComparison
# )
# from .baseline_models import BaselineModelComparison  # 暂时注释掉
# from .plot.simple_plots import training_curve, feature_importance, geneset_network, model_comparison  # 暂时注释掉
# from .cross_validation import (  # 暂时注释掉
#     cross_validation_hyperparameter_search,
#     train_final_model_with_cv,
#     run_cv_experiment,
#     save_cv_results,
#     plot_cv_results,
#     HyperparameterOptimizer,
#     MultiModelCrossValidator,
#     run_multi_model_cv_experiment
# )

# 日志相关
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
    # "build_classification_models",  # 暂时注释掉
    # "train_classification_models",
    # "train_clf_models",
    "compare_models",
    # "ClassificationModelComparison",  # 暂时注释掉
    # "BaselineModelComparison",  # 暂时注释掉
    # "training_curve",  # 暂时注释掉
    # "feature_importance",
    # "geneset_network",
    # "model_comparison",
    # "cross_validation_hyperparameter_search",  # 暂时注释掉
    # "train_final_model_with_cv",
    # "run_cv_experiment",
    # "save_cv_results",
    # "plot_cv_results",
    # "HyperparameterOptimizer",
    # "MultiModelCrossValidator",
    # "run_multi_model_cv_experiment",
    "setup_logging",
    "get_logger",
    "logger"
]