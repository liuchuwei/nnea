"""
NNEA Model Module
Contains model building, training, evaluation and interpretation functionality
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