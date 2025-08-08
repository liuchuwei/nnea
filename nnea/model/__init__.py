"""
NNEA模型模块
包含模型构建、训练、评估和解释功能
"""

from .models import (
    build, train, eval, explain, save_model, load_project, 
    get_summary, train_classification_models, compare_models,
    print_model_structure
)
from .base import BaseModel
from .nnea_model import NNEAModel
from .nnea_classifier import NNEAClassifier
from .nnea_layers import (
    GeneSetLayer, TrainableGeneSetLayer, AttentionLayer, 
    ExplainableLayer, BiologicalConstraintLayer
)

__all__ = [
    "build", "train", "eval", "explain", "save_model", "load_project", 
    "get_summary", "train_classification_models", "compare_models",
    "print_model_structure",
    "BaseModel", "NNEAClassifier", "NNEAModel",
    "GeneSetLayer", "TrainableGeneSetLayer", "AttentionLayer", 
    "ExplainableLayer", "BiologicalConstraintLayer"
] 