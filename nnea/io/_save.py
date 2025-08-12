import os
import torch
import logging
from typing import Dict, Any, Optional
# Avoid circular imports, use type annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._nadata import nadata

# Get logger
logger = logging.getLogger(__name__)


def save_project(nadata_obj, filepath: str, save_data: bool = True) -> None:
    """
    Save nadata project to file
    
    Parameters:
    -----------
    nadata_obj : nadata
        nadata object to save
    filepath : str
        Save path
    save_data : bool
        Whether to save data, if False only save model and configuration
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare data to save
    checkpoint = {}
    
    # Save configuration (get from Model container)
    config = nadata_obj.Model.get_config()
    if config:
        checkpoint['config'] = config
    
    # Save core data (optional)
    if save_data:
        if nadata_obj.X is not None:
            checkpoint['X'] = nadata_obj.X
        if nadata_obj.Meta is not None:
            checkpoint['Meta'] = nadata_obj.Meta
        if nadata_obj.Var is not None:
            checkpoint['Var'] = nadata_obj.Var
        if nadata_obj.Prior is not None:
            checkpoint['Prior'] = nadata_obj.Prior
    
    # Save model information (get from Model container)
    if nadata_obj.Model:
        # Save state dictionaries of all models
        model_states = {}
        for model_name, model in nadata_obj.Model.models.items():
            if hasattr(model, 'state_dict'):
                model_states[model_name] = model.state_dict()
            elif hasattr(model, 'get_params'):
                # For sklearn models, save parameters
                model_states[model_name] = model.get_params()
        
        if model_states:
            checkpoint['model_states'] = model_states
        
        # Training history
        train_results = nadata_obj.Model.get_train_results()
        if train_results:
            checkpoint['train_results'] = train_results
        
        # Data indices
        indices = nadata_obj.Model.get_indices()
        if any(indices.values()):
            checkpoint['indices'] = indices
        
        # Metadata
        metadata = nadata_obj.Model.get_metadata()
        if metadata:
            checkpoint['metadata'] = metadata
    
    # Save to file - ensure compatibility
    torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
    
    logger.info(f"Project saved: {filepath}")
    logger.info(f"Saved data: {list(checkpoint.keys())}")


def save_model_only(nadata_obj, filepath: str) -> None:
    """
    Save only model, not data
    
    Parameters:
    -----------
    nadata_obj : nadata
        nadata object to save
    filepath : str
        Save path
    """
    save_project(nadata_obj, filepath, save_data=False)


def save_data_only(nadata_obj, filepath: str) -> None:
    """
    Save only data, not model
    
    Parameters:
    -----------
    nadata_obj : nadata
        nadata object to save
    filepath : str
        Save path
    """
    # Prepare data to save
    checkpoint = {}
    
    # Save core data
    if nadata_obj.X is not None:
        checkpoint['X'] = nadata_obj.X
    if nadata_obj.Meta is not None:
        checkpoint['Meta'] = nadata_obj.Meta
    if nadata_obj.Var is not None:
        checkpoint['Var'] = nadata_obj.Var
    if nadata_obj.Prior is not None:
        checkpoint['Prior'] = nadata_obj.Prior
    
    # Save data indices
    indices = nadata_obj.Model.get_indices()
    if any(indices.values()):
        checkpoint['indices'] = indices
    
    # Save to file
    torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
    
    logger.info(f"Data saved: {filepath}")


def save_checkpoint(nadata_obj, filepath: str, epoch: int, 
                   save_data: bool = False, **kwargs) -> None:
    """
    Save checkpoint
    
    Parameters:
    -----------
    nadata_obj : nadata
        nadata object to save
    filepath : str
        Save path
    epoch : int
        Current epoch
    save_data : bool
        Whether to save data
    **kwargs : Other parameters
    """
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'config': nadata_obj.Model.get_config(),
        'train_results': nadata_obj.Model.get_train_results(),
        'indices': nadata_obj.Model.get_indices(),
        'metadata': nadata_obj.Model.get_metadata()
    }
    
    # Save model states
    model_states = {}
    for model_name, model in nadata_obj.Model.models.items():
        if hasattr(model, 'state_dict'):
            model_states[model_name] = model.state_dict()
    
    if model_states:
        checkpoint['model_states'] = model_states
    
    # Save core data (optional)
    if save_data:
        if nadata_obj.X is not None:
            checkpoint['X'] = nadata_obj.X
        if nadata_obj.Meta is not None:
            checkpoint['Meta'] = nadata_obj.Meta
        if nadata_obj.Var is not None:
            checkpoint['Var'] = nadata_obj.Var
        if nadata_obj.Prior is not None:
            checkpoint['Prior'] = nadata_obj.Prior
    
    # Add extra parameters
    checkpoint.update(kwargs)
    
    # Save to file
    torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
    
    logger.info(f"Checkpoint saved: {filepath} (epoch {epoch})")


def export_results(nadata_obj, output_dir: str) -> None:
    """
    Export results to specified directory
    
    Parameters:
    -----------
    nadata_obj : nadata
        nadata object to export
    output_dir : str
        Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export training results
    train_results = nadata_obj.Model.get_train_results()
    if train_results:
        import json
        with open(os.path.join(output_dir, 'train_results.json'), 'w') as f:
            json.dump(train_results, f, indent=2, default=str)
    
    # Export configuration
    config = nadata_obj.Model.get_config()
    if config:
        import json
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    # Export model comparison results
    try:
        from ..model.models import compare_models
        comparison_results = compare_models(nadata_obj, verbose=0)
        if 'comparison_df' in comparison_results and comparison_results['comparison_df'] is not None:
            comparison_results['comparison_df'].to_csv(
                os.path.join(output_dir, 'model_comparison.csv'), index=False
            )
    except Exception as e:
        logger.warning(f"Failed to export model comparison results: {e}")
    
    # Export data summary
    try:
        from ..model.models import get_summary
        summary = get_summary(nadata_obj)
        import json
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to export summary: {e}")
    
    logger.info(f"Results exported to: {output_dir}") 