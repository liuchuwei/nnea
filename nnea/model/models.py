"""
NNEA Model Factory
Select different model types based on configuration
"""

import logging
import torch
import os
from typing import Dict, Any, Optional
from .base import BaseModel
from .nnea_classifier import NNEAClassifier
from .nnea_survival import NNEASurvival
from ..utils.helpers import ensure_reproducibility
import torch.nn as nn
import numpy as np
from .nnea_autoencoder import NNEAAutoencoder
from .nnea_regresser import NNEARegresser
from .nnea_umap import NNEAUMAP

logger = logging.getLogger(__name__)

def build_model(config: Dict[str, Any]) -> BaseModel:
    """
    Build model based on configuration
    
    Args:
        config: Model configuration
        
    Returns:
        Built model instance
    """
    # Ensure experiment reproducibility
    ensure_reproducibility(config)
    
    model_type = config.get('global', {}).get('model', 'nnea')
    
    # Ensure device configuration is correctly passed
    device_config = config.get('global', {}).get('device', 'cpu')
    if device_config == 'cuda' and torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        config['device'] = 'cpu'
    
    # Handle NNEA configuration flattening
    if model_type == 'nnea' and 'nnea' in config:
        # Flatten NNEA configuration
        # Directly use nested configuration structure, no flattening
        model_config = config
    else:
        model_config = config
    
    if model_type == 'nnea_classifier':
        logger.info("Building NNEA Classifier")
        return NNEAClassifier(model_config)
    elif model_type == 'nnea_regression':
        logger.info("Building NNEA Regressor")
        return NNEARegresser(model_config)
    elif model_type == 'nnea_survival':
        logger.info("Building NNEA Survival Analysis Model")
        return NNEASurvival(model_config)
    elif model_type == 'nnea_autoencoder':
        logger.info("Building NNEA Autoencoder")
        return NNEAAutoencoder(model_config)
    elif model_type == 'nnea_umap':
        logger.info("Building NNEA UMAP Model")
        return NNEAUMAP(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def build(nadata) -> None:
    """
    Build model and add to nadata's Model container
    
    Args:
        nadata: nadata object
    """
    if nadata is None:
        raise ValueError("nadata object cannot be empty")
    
    # Get model configuration
    config = nadata.Model.get_config()
    if not config:
        # If no configuration, try to get from nadata.config (backward compatibility)
        config = getattr(nadata, 'config', {})
        if config:
            nadata.Model.set_config(config)
    
    model_type = config.get('global', {}).get('model', 'nnea')
    
    # Build model
    model = build_model(config)
    
    # Build model
    model.build(nadata)
    
    # Print model structure information
    if model_type == 'nnea' and hasattr(model, 'model'):
        print_model_structure(model.model)
    
    # Save to nadata's Model container
    nadata.Model.add_model(model_type, model)
    
    logger.info(f"Model built and added to nadata.Model: {model_type}")

def print_model_structure(model):
    """
    Print NNEA model structure information, particularly geneset_layer and focus_layer
    
    Args:
        model: NNEAModel instance
    """
    print("\n" + "="*60)
    print("🔍 NNEA Model Structure Analysis")
    print("="*60)
    
    # Print geneset_layer structure
    if hasattr(model, 'geneset_layer'):
        print("\n📊 Geneset Layer Structure:")
        print("-" * 40)
        geneset_layer = model.geneset_layer
        print(f"Type: {type(geneset_layer).__name__}")
        print(f"Number of genes: {geneset_layer.num_genes}")
        print(f"Number of gene sets: {geneset_layer.num_sets}")
        print(f"Minimum gene set size: {geneset_layer.min_set_size}")
        print(f"Maximum gene set size: {geneset_layer.max_set_size}")
        print(f"Prior knowledge: {'Yes' if geneset_layer.piror_knowledge is not None else 'No'}")
        print(f"Freeze prior: {geneset_layer.freeze_piror}")
        print(f"Dropout rate: {geneset_layer.geneset_dropout.p}")
        
        # Print shape of gene set membership matrix
        if hasattr(geneset_layer, 'set_membership'):
            membership_shape = geneset_layer.set_membership.shape
            print(f"Gene set membership matrix shape: {membership_shape}")
            
            # Calculate sparsity
            membership = geneset_layer.set_membership.detach()
            sparsity = (membership == 0).float().mean().item()
            print(f"Membership matrix sparsity: {sparsity:.3f}")
    
    # Print focus_layer structure
    if hasattr(model, 'focus_layer'):
        print("\n🎯 Focus Layer Structure:")
        print("-" * 40)
        focus_layer = model.focus_layer
        print(f"Type: {type(focus_layer).__name__}")
        
        # Analyze composition of focus_layer
        if isinstance(focus_layer, nn.Sequential):
            print(f"Number of layers: {len(focus_layer)}")
            for i, layer in enumerate(focus_layer):
                print(f"  Layer {i+1}: {type(layer).__name__}")
                if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                    print(f"    Input dimension: {layer.in_features}")
                    print(f"    Output dimension: {layer.out_features}")
        else:
            print(f"Layer structure: {focus_layer}")
    
    # Print biological constraint layer
    if hasattr(model, 'bio_constraint_layer') and model.bio_constraint_layer is not None:
        print("\n🧬 Biological Constraint Layer:")
        print("-" * 40)
        bio_layer = model.bio_constraint_layer
        print(f"Type: {type(bio_layer).__name__}")
        print(f"Input dimension: {bio_layer.input_dim}")
        print(f"Prior knowledge shape: {bio_layer.piror_knowledge.shape}")
    
    # Print overall model information
    print("\n📈 Overall Model Information:")
    print("-" * 40)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params:,}")
    print(f"Number of trainable parameters: {trainable_params:,}")
    print(f"Device: {next(model.parameters()).device}")
    
    print("\n" + "="*60)

def train(nadata, model_name: Optional[str] = None, verbose: int = 1) -> Dict[str, Any]:
    """
    Train model
    
    Args:
        nadata: nadata object
        model_name: Model name, if None, use default model
        verbose: Verbosity, 0=only show progress bar, 1=show basic info, 2=show detailed evaluation results
        
    Returns:
        Training results
    """
    if not nadata.Model.models:
        raise ValueError("nadata.Model has no models, please call build() first")

    # Determine which model to train
    if model_name is None:
        model_type = nadata.Model.get_config().get('global', {}).get('model', 'nnea')
        model = nadata.Model.get_model(model_type)
    else:
        model = nadata.Model.get_model(model_name)

    if model is None:
        raise ValueError(f"Model not found: {model_name or 'default'}")

    # Check if tailor strategy is enabled
    tailor_enabled = nadata.Model.get_config().get('training', {}).get('tailor', False)
    
    if tailor_enabled:
        # Use tailor training strategy
        train_results = _train_with_tailor(nadata, model, verbose=verbose)
    else:
        # Use standard training strategy
        train_results = model.train(nadata, verbose=verbose)
    
    # Save training results to Model container
    nadata.Model.set_train_results(train_results)
    
    return train_results

def _train_with_tailor(nadata, model, verbose: int = 1) -> Dict[str, Any]:
    """
    Train model using loop tailor strategy, model pruning every tailor_epoch epochs
    Add early stopping mechanism: if validation loss does not decrease for 3 consecutive tailor epochs, stop training
    
    Args:
        nadata: nadata object
        model: Model instance
        verbose: Verbosity
        
    Returns:
        Training results
    """
    config = nadata.Model.get_config()
    training_config = config.get('training', {})
    
    # Get tailor related parameters
    tailor_epoch = training_config.get('tailor_epoch', 20)
    tailor_geneset = training_config.get('tailor_geneset', 2)
    total_epochs = training_config.get('epochs', 100)
    
    # Get output directory (created in set_config)
    outdir = config.get('global', {}).get('outdir', 'experiment/test')
    
    logger = logging.getLogger(__name__)
    logger.info(f"Enabling loop tailor strategy: tailor_epoch={tailor_epoch}, tailor_geneset={tailor_geneset}, total_epochs={total_epochs}")
    logger.info(f"Output directory: {outdir}")
    
    # Initialize variables
    current_model = model
    current_epoch = 0
    stage_results = []
    tailor_history = []
    model_type = config.get('global', {}).get('model', 'nnea')
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_stage = 0
    no_improvement_count = 0
    max_no_improvement = 3  # Stop if validation loss does not decrease for 3 consecutive tailor epochs
    
    # Loop training and pruning
    while current_epoch < total_epochs:
        # Calculate number of epochs for current stage
        if current_epoch + tailor_epoch <= total_epochs:
            epochs_to_train = tailor_epoch
        else:
            epochs_to_train = total_epochs - current_epoch
        
        stage_num = len(stage_results) + 1
        logger.info(f"Stage {stage_num} training: from epoch {current_epoch} to epoch {current_epoch + epochs_to_train}")
        
        # Train current stage
        stage_result = current_model.train(nadata, verbose=verbose, max_epochs=epochs_to_train, continue_training=(current_epoch > 0))
        stage_results.append(stage_result)
        
        current_epoch += epochs_to_train
        
        # Get validation loss for current stage
        current_val_loss = stage_result.get('final_val_loss', float('inf'))
        if current_val_loss is None:
            current_val_loss = float('inf')
        
        logger.info(f"Stage {stage_num} validation loss: {current_val_loss:.6f}")
        
        # Check if it's the best model
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_stage = stage_num
            no_improvement_count = 0
            logger.info(f"✅ Stage {stage_num} validation loss improved to {best_val_loss:.6f}")
        else:
            no_improvement_count += 1
            logger.info(f"⚠️ Stage {stage_num} validation loss not improved, consecutive no improvement count: {no_improvement_count}/{max_no_improvement}")
        
        # Save current stage results
        stage_info = {
            'stage': stage_num,
            'epoch': current_epoch,
            'val_loss': current_val_loss,
            'best_val_loss': best_val_loss,
            'no_improvement_count': no_improvement_count
        }
        tailor_history.append(stage_info)
        
        logger.info(f"📊 Stage {stage_num} training completed, validation loss: {current_val_loss:.6f}")
        
        # Check early stopping condition
        if no_improvement_count >= max_no_improvement:
            logger.info(f"🛑 Early stopping triggered after {max_no_improvement} consecutive tailor epochs with no improvement in validation loss!")
            logger.info(f"    Best validation loss: {best_val_loss:.6f} (Stage {best_stage})")
            break
        
        # If not yet at total epochs, perform pruning
        if current_epoch <= total_epochs:
            logger.info(f"Stage {stage_num} training completed, starting model pruning...")
            
            # Get gene set importance
            logger.info("Calculating gene set importance...")
            try:
                explain_results = current_model.explain(nadata, method='importance')
                geneset_importance = np.array(explain_results['importance']['geneset_importance'])
                logger.info(f"Gene set importance calculation completed, shape: {geneset_importance.shape}")
            except Exception as e:
                logger.error(f"Gene set importance calculation failed: {e}")
                # Use random importance as fallback
                geneset_importance = np.random.rand(current_model.model.num_genesets)
            
            # Determine gene sets to remove (least important)
            num_genesets_to_remove = tailor_geneset
            if num_genesets_to_remove >= len(geneset_importance):
                logger.warning(f"Number of gene sets to remove ({num_genesets_to_remove}) is greater than or equal to total gene sets ({len(geneset_importance)}), adjusting to remove 1")
                num_genesets_to_remove = 1
            
            # Find indices of least important gene sets
            least_important_indices = np.argsort(geneset_importance)[:num_genesets_to_remove]
            important_indices = np.argsort(geneset_importance)[num_genesets_to_remove:]
            
            # Try to get genesets_annotated key
            genesets_annotated = nadata.uns.get('nnea_explain', {}).get('importance', {}).get('genesets_annotated', {})
            if genesets_annotated:
                # Get list of geneset keys
                geneset_keys = list(genesets_annotated.keys())
                
                # Map indices to geneset keys
                removed_keys = [geneset_keys[idx] if idx < len(geneset_keys) else f"Geneset_{idx}" for idx in least_important_indices]
                kept_keys = [geneset_keys[idx] if idx < len(geneset_keys) else f"Geneset_{idx}" for idx in important_indices]

                logger.info(f"Removing gene sets: {removed_keys}")
                logger.info(f"Keeping gene sets: {kept_keys}")
            else:
                logger.info(f"Removing gene set indices: {least_important_indices.tolist()}")
                logger.info(f"Keeping gene set indices: {important_indices.tolist()}")
            
            # Record pruning information
            tailor_info = {
                'stage': stage_num,
                'epoch': current_epoch,
                'removed_genesets': least_important_indices.tolist(),
                'kept_genesets': important_indices.tolist(),
                'geneset_importance': geneset_importance.tolist(),
                'num_genesets_before': len(geneset_importance),
                'num_genesets_after': len(important_indices),
                'val_loss': current_val_loss,
                'no_improvement_count': no_improvement_count
            }
            
            # If genesets_annotated, add key information
            if genesets_annotated:
                geneset_keys = list(genesets_annotated.keys())
                removed_keys = [geneset_keys[idx] if idx < len(geneset_keys) else f"Geneset_{idx}" for idx in least_important_indices]
                kept_keys = [geneset_keys[idx] if idx < len(geneset_keys) else f"Geneset_{idx}" for idx in important_indices]
                tailor_info['removed_geneset_keys'] = removed_keys
                tailor_info['kept_geneset_keys'] = kept_keys
            tailor_history.append(tailor_info)
            
            # Prune model
            logger.info("Starting model pruning...")
            cropped_model = _crop_nnea_model(nadata, current_model, important_indices, config)
            
            # Update model in nadata
            nadata.Model.add_model(f"{model_type}_stage_{stage_num}", current_model)
            
            # Update current model to the cropped model
            current_model = cropped_model
            
            logger.info(f"Stage {stage_num} pruning completed, remaining gene sets: {len(important_indices)}")
        else:
            logger.info("Training completed, no further pruning needed")
    
    # Load best model from nadata Model container
    if best_stage is not None:
        logger.info(f"🔄 Loading best model (Stage {best_stage}, validation loss: {best_val_loss:.6f})")
        
        # Try to get the best stage model from nadata Model container
        best_stage_model_name = f"{model_type}_stage_{best_stage}"
        if nadata.Model.has_model(best_stage_model_name):
            best_model = nadata.Model.get_model(best_stage_model_name)
            current_model = best_model
            logger.info(f"✅ Successfully loaded best model from stage {best_stage}")
        else:
            logger.warning(f"⚠️ Best stage model '{best_stage_model_name}' not found in nadata Model container")
            logger.warning("⚠️ Using current model as final model")
        
        # Update model in nadata to best model
        nadata.Model.add_model(f"{model_type}", current_model)
        
        # Save final best model
        # final_best_model_path = os.path.join(outdir, "best_model_final.pth")
        # torch.save(current_model.model.state_dict(), final_best_model_path)
        #
        # final_best_nadata_path = os.path.join(outdir, "best_nadata_final.pkl")
        # try:
        #     nadata.save(final_best_nadata_path, format="pickle", save_data=True)
        #     logger.info(f"💾 Final best model saved to: {final_best_model_path}")
        #     logger.info(f"💾 Final best nadata saved to: {final_best_nadata_path}")
        # except Exception as e:
        #     logger.error(f"Failed to save final best model: {e}")
    else:
        logger.warning("⚠️ Best stage not found, using current model")
        # Save current model as final model
        # final_model_path = os.path.join(outdir, "final_model.pth")
        # torch.save(current_model.model.state_dict(), final_model_path)
        # logger.info(f"💾 Current model saved to: {final_model_path}")
    
    # Combine training results
    combined_results = {
        'stage_results': stage_results,
        'tailor_history': tailor_history,
        'tailor_info': {
            'tailor_epoch': tailor_epoch,
            'tailor_geneset': tailor_geneset,
            'total_stages': len(stage_results),
            'final_geneset_count': current_model.model.num_genesets if hasattr(current_model.model, 'num_genesets') else 'unknown',
            'best_stage': best_stage,
            'best_val_loss': best_val_loss,
            'early_stopped': no_improvement_count >= max_no_improvement,
            'no_improvement_count': no_improvement_count
        }
    }
    
    logger.info(f"Loop Tailor strategy training completed, {len(stage_results)} stages performed")
    if no_improvement_count >= max_no_improvement:
        logger.info(f"Training ended due to early stopping, best model from Stage {best_stage}")
    else:
        logger.info(f"Training completed normally, best model from Stage {best_stage}")
    
    return combined_results

def _crop_nnea_model(nadata, model, important_indices: np.ndarray, config: Dict[str, Any]):
    """
    Prune NNEA model, remove unimportant gene sets
    
    Args:
        nadata: nadata object
        model: Original model
        important_indices: Indices of gene sets to keep
        config: Model configuration
        
    Returns:
        Pruned model
    """
    logger = logging.getLogger(__name__)
    
    # Create new configuration
    cropped_config = config.copy()
    
    # Update number of gene sets
    nnea_config = cropped_config.get('nnea', {})
    geneset_config = nnea_config.get('geneset_layer', {})
    original_num_genesets = geneset_config.get('num_genesets', 20)
    new_num_genesets = len(important_indices)
    
    geneset_config['num_genesets'] = new_num_genesets
    nnea_config['geneset_layer'] = geneset_config
    cropped_config['nnea'] = nnea_config
    
    logger.info(f"Gene set count pruned: {original_num_genesets} -> {new_num_genesets}")
    
    # Create new model instance
    if cropped_config.get('global').get("model") == 'nnea_classifier':
        from .nnea_classifier import NNEAClassifier
        cropped_model = NNEAClassifier(cropped_config)
    elif cropped_config.get('global').get('model') == 'nnea_survival':
        from .nnea_survival import NNEASurvival
        cropped_model = NNEASurvival(cropped_config)
    elif cropped_config.get('global').get('model') == 'nnea_regression':
        from .nnea_regresser import NNEARegresser
        cropped_model = NNEARegresser(cropped_config)
    # Build new model
    cropped_model.build(nadata)
    
    # Get gene set layer parameters from original model
    original_geneset_params = model.model.geneset_layer.get_geneset_parameters()
    
    # Set gene set layer parameters for the pruned model
    important_indices_tensor = torch.tensor(important_indices, dtype=torch.long)
    cropped_model.model.geneset_layer.set_geneset_parameters(original_geneset_params, important_indices_tensor)

    # Set device
    cropped_model.device = model.device
    cropped_model.model = cropped_model.model.to(cropped_model.device)
    
    logger.info("Model pruning completed")
    return cropped_model


def _update_focus_layer_parameters(original_model, cropped_model, important_indices: np.ndarray):
    """
    Update focus_layer parameters to adapt to the change in geneset_layer output dimensions
    
    Args:
        original_model: Original model
        cropped_model: Pruned model
        important_indices: Indices of gene sets to keep
    """
    logger = logging.getLogger(__name__)
    
    # Get original and pruned focus_layer
    original_focus_layer = original_model.focus_layer
    cropped_focus_layer = cropped_model.focus_layer
    
    if not isinstance(original_focus_layer, nn.Sequential) or not isinstance(cropped_focus_layer, nn.Sequential):
        logger.warning("focus_layer is not a Sequential structure, skipping parameter update")
        return
    
    # Find the first Linear layer (usually the first layer of focus_layer)
    first_linear_layer = None
    first_linear_idx = None
    
    for i, layer in enumerate(original_focus_layer):
        if isinstance(layer, nn.Linear):
            first_linear_layer = layer
            first_linear_idx = i
            break
    
    if first_linear_layer is None:
        logger.warning("Linear layer not found, skipping focus_layer parameter update")
        return
    
    # Get the first Linear layer of the pruned model
    cropped_first_linear = cropped_focus_layer[first_linear_idx]
    
    if not isinstance(cropped_first_linear, nn.Linear):
        logger.warning("The first layer of the pruned model is not a Linear layer, skipping parameter update")
        return
    
    # Check if dimensions match
    original_input_dim = first_linear_layer.in_features
    cropped_input_dim = cropped_first_linear.in_features
    
    if original_input_dim != len(important_indices):
        logger.warning(f"Dimension mismatch: Original input dimension {original_input_dim}, number of important indices {len(important_indices)}")
        return
    
    # Update weights: only keep weights corresponding to important gene sets
    original_weight = first_linear_layer.weight.data
    original_bias = first_linear_layer.bias.data if first_linear_layer.bias is not None else None
    
    # Select rows of weights corresponding to important gene sets
    important_indices_tensor = torch.tensor(important_indices, dtype=torch.long, device=original_weight.device)
    cropped_weight = original_weight[:, important_indices_tensor]
    
    # Update weights of the pruned model
    cropped_first_linear.weight.data = cropped_weight
    
    # Bias term remains unchanged (if any)
    if original_bias is not None and cropped_first_linear.bias is not None:
        cropped_first_linear.bias.data = original_bias.clone()
    
    logger.info(f"focus_layer first layer parameter update completed: Input dimension {original_input_dim} -> {cropped_input_dim}")

def eval(nadata, split='test', model_name: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate model
    
    Args:
        nadata: nadata object
        split: Data split for evaluation
        model_name: Model name
        
    Returns:
        Evaluation results
    """
    if not nadata.Model.models:
        raise ValueError("nadata.Model has no models, please call build() first")
    
    # Determine which model to evaluate
    if model_name is None:
        model_type = nadata.Model.get_config().get('global', {}).get('model', 'nnea')
        model = nadata.Model.get_model(model_type)
    else:
        model = nadata.Model.get_model(model_name)
    
    if model is None:
        raise ValueError(f"Model not found: {model_name or 'default'}")
    
    # Evaluate model
    return model.evaluate(nadata, split)

def explain(nadata, method='importance', model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Explain model
    
    Args:
        nadata: nadata object
        method: Explanation method
        model_name: Model name
        
    Returns:
        Explanation results
    """
    if not nadata.Model.models:
        raise ValueError("nadata.Model has no models, please call build() first")
    
    # Determine which model to explain
    if model_name is None:
        model_type = nadata.Model.get_config().get('global', {}).get('model', 'nnea')
        model = nadata.Model.get_model(model_type)
    else:
        model = nadata.Model.get_model(model_name)
    
    if model is None:
        raise ValueError(f"Model not found: {model_name or 'default'}")
    
    # Explain model
    return model.explain(nadata, method)

def save_model(nadata, save_path: str, model_name: Optional[str] = None) -> None:
    """
    Save model or entire nadata project
    
    Args:
        nadata: nadata object
        save_path: Save path
        model_name: Model name, if None, save entire project
    """
    if model_name is None:
        # Save entire project
        nadata.save(save_path)
        logger.info(f"Project saved to: {save_path}")
    else:
        # Save specific model
        model = nadata.Model.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")
        
        # Save model state
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model {model_name} saved to: {save_path}")

def load_project(load_path: str):
    """
    Load nadata project
    
    Args:
        load_path: Load path
        
    Returns:
        nadata object
    """
    from ..io._load import load_project as load_project_impl
    return load_project_impl(load_path)

def get_summary(nadata) -> Dict[str, Any]:
    """
    Get nadata summary information
    
    Args:
        nadata: nadata object
        
    Returns:
        Summary information dictionary
    """
    summary = {
        'data_info': {},
        'model_info': {},
        'config_info': {}
    }
    
    # Data information
    if nadata.X is not None:
        summary['data_info']['X_shape'] = nadata.X.shape
    if nadata.Meta is not None:
        summary['data_info']['Meta_shape'] = nadata.Meta.shape
    if nadata.Var is not None:
        summary['data_info']['Var_shape'] = nadata.Var.shape
    if nadata.Prior is not None:
        summary['data_info']['Prior_shape'] = nadata.Prior.shape
    
    # Model information
    summary['model_info']['models'] = nadata.Model.list_models()
    summary['model_info']['config_keys'] = list(nadata.Model.get_config().keys())
    summary['model_info']['train_results_keys'] = list(nadata.Model.get_train_results().keys())
    
    # Configuration information
    config = nadata.Model.get_config()
    if config:
        summary['config_info']['model_type'] = config.get('global', {}).get('model', 'unknown')
        summary['config_info']['task'] = config.get('global', {}).get('task', 'unknown')
        summary['config_info']['device'] = config.get('global', {}).get('device', 'cpu')
    
    return summary

def train_classification_models(nadata, config: Optional[Dict[str, Any]] = None, verbose: int = 1) -> Dict[str, Any]:
    """
    Train classification models
    
    Args:
        nadata: nadata object
        config: Configuration dictionary
        verbose: Verbosity
        
    Returns:
        Training results
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import numpy as np
    
    # Get configuration
    if config is None:
        config = nadata.Model.get_config()
    
    # Get data
    X_train = nadata.X[:, nadata.Model.get_indices('train')]
    X_test = nadata.X[:, nadata.Model.get_indices('test')]
    
    # Get target column name
    target_column = 'class'  # Default to 'class'
    if config and 'dataset' in config:
        target_column = config['dataset'].get('target_column', 'class')
    
    y_train = nadata.Meta.iloc[nadata.Model.get_indices('train')][target_column].values
    y_test = nadata.Meta.iloc[nadata.Model.get_indices('test')][target_column].values
    
    # Define models to train
    models_to_train = config.get('classification', {}).get('models', ['logistic_regression', 'random_forest'])
    
    results = {
        'models': {},
        'comparison_df': None
    }
    
    for model_name in models_to_train:
        if verbose >= 1:
            print(f"Training {model_name}...")
        
        if model_name == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'support_vector_machine':
            model = SVC(probability=True, random_state=42)
        else:
            if verbose >= 1:
                print(f"Skipping unknown model: {model_name}")
            continue
        
        # Train model
        model.fit(X_train.T, y_train)
        
        # Predict
        y_pred = model.predict(X_test.T)
        y_pred_proba = model.predict_proba(X_test.T)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Save results
        results['models'][model_name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Add to nadata's Model container
        nadata.Model.add_model(f'classification_{model_name}', model)
        
        if verbose >= 1:
            print(f"  {model_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    # Create comparison DataFrame
    if results['models']:
        comparison_data = []
        for name, result in results['models'].items():
            comparison_data.append({
                'model': name,
                'accuracy': result['accuracy'],
                'auc': result['auc']
            })
        
        import pandas as pd
        results['comparison_df'] = pd.DataFrame(comparison_data)
    
    return results

def compare_models(nadata, config: Optional[Dict[str, Any]] = None, verbose: int = 1) -> Dict[str, Any]:
    """
    Compare performance of different models
    
    Args:
        nadata: nadata object
        config: Configuration dictionary
        verbose: Verbosity
        
    Returns:
        Comparison results
    """
    # Get all models
    all_models = nadata.Model.list_models()
    
    if verbose >= 1:
        print(f"Comparing {len(all_models)} models: {all_models}")
    
    results = {
        'models': {},
        'comparison_df': None
    }
    
    # Get test data
    test_indices = nadata.Model.get_indices('test')
    if test_indices is None:
        raise ValueError("Test set indices not found")
    
    X_test = nadata.X[:, test_indices]
    
    # Get target column name
    target_column = nadata.Model.get_config().get('dataset', {}).get('target_column', 'class')
    y_test = nadata.Meta.iloc[test_indices][target_column].values
    
    for model_name in all_models:
        model = nadata.Model.get_model(model_name)
        if model is None:
            continue
        
        if verbose >= 1:
            print(f"Evaluating {model_name}...")
        
        try:
            # For NNEA models
            if hasattr(model, 'evaluate'):
                eval_result = model.evaluate(nadata, 'test')
                results['models'][model_name] = eval_result
            else:
                # For sklearn models
                y_pred = model.predict(X_test.T)
                y_pred_proba = model.predict_proba(X_test.T)[:, 1]
                
                from sklearn.metrics import accuracy_score, roc_auc_score
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results['models'][model_name] = {
                    'accuracy': accuracy,
                    'auc': auc
                }
            
            if verbose >= 1:
                if 'accuracy' in results['models'][model_name]:
                    print(f"  {model_name} - Accuracy: {results['models'][model_name]['accuracy']:.4f}")
                if 'auc' in results['models'][model_name]:
                    print(f"  {model_name} - AUC: {results['models'][model_name]['auc']:.4f}")
                    
        except Exception as e:
            if verbose >= 1:
                print(f"   Evaluation of {model_name} failed: {e}")
            continue
    
    # Create comparison DataFrame
    if results['models']:
        comparison_data = []
        for name, result in results['models'].items():
            row = {'model': name}
            if 'accuracy' in result:
                row['accuracy'] = result['accuracy']
            if 'auc' in result:
                row['auc'] = result['auc']
            comparison_data.append(row)
        
        import pandas as pd
        results['comparison_df'] = pd.DataFrame(comparison_data)
    
    return results

def predict(nadata, split='test', model_name: Optional[str] = None, return_probabilities: bool = True) -> Dict[str, Any]:
    """
    Model prediction

    Args:
        nadata: nadata object
        split: Data split for prediction
        model_name: Model name
        return_probabilities: Whether to return probability values

    Returns:
        Prediction results dictionary, including predicted values, probability values, true labels, etc.
    """
    from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
    import torch

    logger = logging.getLogger(__name__)

    if not nadata.Model.models:
        raise ValueError("nadata.Model has no models, please call build() first")

    # Determine which model to predict
    if model_name is None:
        model_type = nadata.Model.get_config().get('global', {}).get('model', 'nnea')
        model = nadata.Model.get_model(model_type)
    else:
        model = nadata.Model.get_model(model_name)

    if model is None:
        raise ValueError(f"Model not found: {model_name or 'default'}")

    # Get data indices
    indices = nadata.Model.get_indices(split)
    if indices is None:
        logger.warning(f"Indices for {split} set not found, cannot perform prediction")
        return {
            'y_test': None,
            'y_pred': None,
            'y_proba': None,
            'predictions': None,
            'error': f"Indices for {split} set not found"
        }

    try:
        # Get test set data
        X_test = nadata.X[indices]  # Transpose to (number of samples, number of features)

        # Get target column name
        config = nadata.Model.get_config()

        logger.info(f"🔮 Performing model prediction...")

        # Model prediction
        model.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(model.device)
            outputs = model.model(X_test_tensor)

        # Process prediction results based on task type
        task_type = getattr(model, 'task', 'classification')
        
        if task_type == 'classification':
            target_col = config.get('dataset', {}).get('target_column', 'target')
            y_test = nadata.Meta.iloc[indices][target_col].values
            logger.info(f"�� Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

            # Classification task handling
            if outputs.shape[1] == 2:
                # Binary classification case
                y_proba = torch.softmax(outputs, dim=1).cpu().numpy()[:, 1]
                y_pred = (y_proba > 0.5).astype(int)
            else:
                # Multi-classification case
                y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                y_pred = np.argmax(y_proba, axis=1)

            # Calculate evaluation metrics
            if len(np.unique(y_test)) == 2:
                # Binary classification
                auc = roc_auc_score(y_test, y_proba)
                logger.info(f"📊 Test set AUC: {auc:.4f}")
            else:
                # Multi-classification
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                logger.info(f"📊 Test set AUC: {auc:.4f}")

            # Output classification report
            logger.info("📊 Prediction results:")
            logger.info(f"Test set classification report:\n{classification_report(y_test, y_pred)}")

            # Save prediction results to Model container
            prediction_results = {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'predictions': outputs.cpu().numpy(),
                'auc': auc,
                'split': split,
                'task_type': 'classification'
            }

        elif task_type == 'survival':
            # Survival task handling
            time_col = config.get('dataset', {}).get('time_column', 'Time')
            event_col = config.get('dataset', {}).get('event_column', 'Event')
            
            times = nadata.Meta.iloc[indices][time_col].values
            events = nadata.Meta.iloc[indices][event_col].values
            
            # Survival analysis prediction results
            risk_scores = outputs.cpu().numpy().flatten()
            
            # Calculate survival analysis metrics
            from lifelines.utils import concordance_index
            c_index = concordance_index(times, -risk_scores, events)
            
            logger.info(f"📊 Test set C-index: {c_index:.4f}")
            
            # Save prediction results to Model container
            prediction_results = {
                'times': times,
                'events': events,
                'risk_scores': risk_scores,
                'predictions': outputs.cpu().numpy(),
                'c_index': c_index,
                'split': split,
                'task_type': 'survival'
            }

        elif task_type == 'regression':
            # Regression task handling
            predictions = outputs.cpu().numpy().flatten()
            
            # Calculate regression evaluation metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mse)
            
            logger.info(f"📊 Test set regression metrics:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            
            # Save prediction results to Model container
            prediction_results = {
                'y_test': y_test,
                'predictions': predictions,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'split': split,
                'task_type': 'regression'
            }

        else:
            # Other task types
            predictions = outputs.cpu().numpy()
            prediction_results = {
                'y_test': y_test,
                'predictions': predictions,
                'split': split,
                'task_type': task_type
            }

        # Save to nadata's Model container
        nadata.Model.add_metadata('prediction_results', prediction_results)

        logger.info(f"✅ Model prediction completed, results saved to nadata.Model")

        return prediction_results

    except Exception as e:
        logger.error(f"❌ Model prediction failed: {e}")
        return {
            'y_test': None,
            'y_pred': None,
            'y_proba': None,
            'predictions': None,
            'error': str(e)
        }