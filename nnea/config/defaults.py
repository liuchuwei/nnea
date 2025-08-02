"""
默认配置模块
提供各种默认配置参数
"""

from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        "model": {
            "type": "neural_network",
            "layers": [100, 50, 25],
            "activation": "relu",
            "dropout": 0.2,
            "batch_norm": True,
            "regularization": "l2",
            "regularization_factor": 0.01
        },
        "data": {
            "train_path": "",
            "test_path": "",
            "validation_split": 0.2,
            "batch_size": 32,
            "shuffle": True,
            "normalize": True,
            "augmentation": False
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "early_stopping": True,
            "patience": 10,
            "reduce_lr_on_plateau": True,
            "min_lr": 1e-7
        },
        "output": {
            "model_path": "models/",
            "results_path": "results/",
            "log_path": "logs/",
            "save_best_only": True,
            "save_format": "h5"
        },
        "explainability": {
            "feature_importance": True,
            "shap_values": True,
            "gradient_analysis": True,
            "attention_weights": True
        }
    }


def get_classification_config() -> Dict[str, Any]:
    """
    获取分类任务默认配置
    
    Returns:
        分类任务配置
    """
    config = get_default_config()
    config["model"]["output_activation"] = "softmax"
    config["training"]["loss"] = "categorical_crossentropy"
    config["training"]["metrics"] = ["accuracy", "precision", "recall", "f1"]
    return config


def get_regression_config() -> Dict[str, Any]:
    """
    获取回归任务默认配置
    
    Returns:
        回归任务配置
    """
    config = get_default_config()
    config["model"]["output_activation"] = "linear"
    config["training"]["loss"] = "mse"
    config["training"]["metrics"] = ["mae", "mse", "rmse"]
    return config


def get_binary_classification_config() -> Dict[str, Any]:
    """
    获取二分类任务默认配置
    
    Returns:
        二分类任务配置
    """
    config = get_default_config()
    config["model"]["output_activation"] = "sigmoid"
    config["training"]["loss"] = "binary_crossentropy"
    config["training"]["metrics"] = ["accuracy", "precision", "recall", "f1", "auc"]
    return config


def get_multi_class_config() -> Dict[str, Any]:
    """
    获取多分类任务默认配置
    
    Returns:
        多分类任务配置
    """
    config = get_default_config()
    config["model"]["output_activation"] = "softmax"
    config["training"]["loss"] = "categorical_crossentropy"
    config["training"]["metrics"] = ["accuracy", "precision", "recall", "f1"]
    return config


def get_transcriptomics_config() -> Dict[str, Any]:
    """
    获取转录组学分析默认配置
    
    Returns:
        转录组学配置
    """
    config = get_default_config()
    config["model"]["type"] = "transcriptomics_nn"
    config["model"]["layers"] = [500, 250, 100, 50]
    config["model"]["dropout"] = 0.3
    config["data"]["normalize"] = True
    config["data"]["feature_selection"] = True
    config["data"]["max_features"] = 1000
    config["explainability"]["gene_importance"] = True
    config["explainability"]["pathway_analysis"] = True
    return config


def get_optimizer_config(optimizer_name: str) -> Dict[str, Any]:
    """
    获取优化器配置
    
    Args:
        optimizer_name: 优化器名称
    
    Returns:
        优化器配置
    """
    optimizers = {
        "adam": {
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-7
        },
        "sgd": {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov": True
        },
        "rmsprop": {
            "learning_rate": 0.001,
            "rho": 0.9,
            "epsilon": 1e-7
        },
        "adagrad": {
            "learning_rate": 0.01,
            "epsilon": 1e-7
        }
    }
    
    return optimizers.get(optimizer_name, optimizers["adam"])


def get_activation_config(activation_name: str) -> Dict[str, Any]:
    """
    获取激活函数配置
    
    Args:
        activation_name: 激活函数名称
    
    Returns:
        激活函数配置
    """
    activations = {
        "relu": {
            "type": "relu",
            "alpha": 0.0,
            "max_value": None,
            "threshold": 0.0
        },
        "leaky_relu": {
            "type": "leaky_relu",
            "alpha": 0.01
        },
        "elu": {
            "type": "elu",
            "alpha": 1.0
        },
        "selu": {
            "type": "selu"
        },
        "tanh": {
            "type": "tanh"
        },
        "sigmoid": {
            "type": "sigmoid"
        }
    }
    
    return activations.get(activation_name, activations["relu"])


def get_loss_config(loss_name: str) -> Dict[str, Any]:
    """
    获取损失函数配置
    
    Args:
        loss_name: 损失函数名称
    
    Returns:
        损失函数配置
    """
    losses = {
        "binary_crossentropy": {
            "type": "binary_crossentropy",
            "from_logits": False
        },
        "categorical_crossentropy": {
            "type": "categorical_crossentropy",
            "from_logits": False
        },
        "mse": {
            "type": "mse"
        },
        "mae": {
            "type": "mae"
        },
        "huber": {
            "type": "huber",
            "delta": 1.0
        }
    }
    
    return losses.get(loss_name, losses["mse"]) 