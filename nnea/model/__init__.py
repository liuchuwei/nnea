"""
模型模块，包含模型构建、训练、评估、解释等功能
"""

from . import base
from . import models
from . import nnea_model
from . import nnea_layers
# from . import classification_models  # 暂时注释掉，因为模块不存在

# 导入主要功能
from .base import BaseModel
from .models import build, train, eval, explain, save_model, load_project, get_summary
# from .classification_models import (  # 暂时注释掉
#     ClassificationModel, LogisticRegressionModel, RandomForestModel, 
#     SVMModel, MLPModel, ClassificationModelComparison,
#     build_classification_models, train_classification_models
# )

__all__ = [
    "BaseModel",
    "build",
    "train", 
    "eval",
    "explain",
    "save_model",
    "load_project",
    "get_summary",
    # "ClassificationModel",  # 暂时注释掉
    # "LogisticRegressionModel", 
    # "RandomForestModel",
    # "SVMModel", 
    # "MLPModel", 
    # "ClassificationModelComparison",
    # "build_classification_models",
    # "train_classification_models"
] 