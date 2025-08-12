import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from nnea.model.nnea_model import NNEAModel
from nnea.model.base import BaseModel
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


class NNEAClassifier(BaseModel):
    """
    NNEA Classifier
    Implements an interpretable classification model with TrainableGeneSetLayer as the core
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NNEA Classifier
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.task = 'classification'
        
    def build(self, nadata) -> None:
        """
        Build model
        
        Args:
            nadata: nadata object
        """
        if nadata is None:
            raise ValueError("nadata object cannot be empty")
        
        # Get input dimension
        if hasattr(nadata, 'X') and nadata.X is not None:
            input_dim = nadata.X.shape[1]  # Number of genes
        else:
            raise ValueError("Expression matrix not loaded")
        
        # Get output dimension
        if hasattr(nadata, 'Meta') and nadata.Meta is not None:
            target_col = self.config.get('dataset', {}).get('target_column', 'target')
            if target_col in nadata.Meta.columns:
                unique_classes = nadata.Meta[target_col].nunique()
                # For classification tasks, output dimension should equal the number of classes
                output_dim = unique_classes
            else:
                output_dim = 2  # Default binary classification
        else:
            output_dim = 2
        
        # Get nnea configuration section
        nnea_config = self.config.get('nnea', {})
        
        # Process prior knowledge
        piror_knowledge = None
        use_piror_knowledge = nnea_config.get('piror_knowledge', {}).get('use_piror_knowledge', False)
        if use_piror_knowledge:
            # Get gene name list
            gene_names = None
            if hasattr(nadata, 'Var') and nadata.Var is not None:
                gene_names = nadata.Var['Gene'].tolist()
                
            if gene_names is not None:
                # Import prior knowledge loading function from nnea.io module
                from nnea.io._load import load_piror_knowledge
                piror_knowledge = load_piror_knowledge(self.config, gene_names)
                
                if piror_knowledge is not None:
                    self.logger.info(f"Successfully loaded prior knowledge, shape: {piror_knowledge.shape}")
                    piror_knowledge = torch.tensor(piror_knowledge, dtype=torch.float32)
                    # Ensure prior knowledge matrix matches input dimension
                    if piror_knowledge.shape[1] != input_dim:
                        self.logger.warning(f"Prior knowledge matrix dimension ({piror_knowledge.shape[1]}) does not match input dimension ({input_dim})")
                        # If dimensions don't match, create random matrix as backup
                        num_genesets = piror_knowledge.shape[0]
                        piror_knowledge = np.random.rand(num_genesets, input_dim)
                        piror_knowledge = (piror_knowledge > 0.8).astype(np.float32)
                else:
                    self.logger.warning("Prior knowledge loading failed, using random matrix")
                    num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
                    piror_knowledge = np.random.rand(num_genesets, input_dim)
                    piror_knowledge = (piror_knowledge > 0.8).astype(np.float32)
            else:
                self.logger.warning("Cannot get gene name list, using random matrix")
                num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
                piror_knowledge = np.random.rand(num_genesets, input_dim)
                piror_knowledge = (piror_knowledge > 0.8).astype(np.float32)
        
        # Process explain_knowledge configuration
        explain_knowledge_path = self.config.get('explain', {}).get('explain_knowledge')
        if explain_knowledge_path:
            # Ensure nadata has uns attribute
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            
            # Save explain_knowledge path to nadata's uns dictionary
            nadata.uns['explain_knowledge_path'] = explain_knowledge_path
            self.logger.info(f"Saved explain_knowledge path to nadata.uns: {explain_knowledge_path}")
        
        # Update configuration
        self.config['input_dim'] = input_dim
        self.config['output_dim'] = output_dim
        self.config['device'] = str(self.device)  # Ensure device configuration is correctly passed
        
        # Update prior knowledge in nnea configuration
        if 'nnea' not in self.config:
            self.config['nnea'] = {}
        if 'piror_knowledge' not in self.config['nnea']:
            self.config['nnea']['piror_knowledge'] = {}
        self.config['nnea']['piror_knowledge']['piror_knowledge'] = piror_knowledge
        
        # Create model
        self.model = NNEAModel(self.config)
        self.model.to(self.device)
        
        # Ensure all model components are on the correct device
        if hasattr(self.model, 'geneset_layer'):
            self.model.geneset_layer.to(self.device)

        self.logger.info(f"NNEA Classifier built: input_dim={input_dim}, output_dim={output_dim}")
        num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
        self.logger.info(f"Number of genesets: {num_genesets}")
        self.logger.info(f"Using prior knowledge: {use_piror_knowledge}")

        self.assist_config = self.config.get('nnea', {}).get('assist_layer', {})
        self.assist_type = self.assist_config.get('type', "rec")

    def train(self, nadata, verbose: int = 1, max_epochs: Optional[int] = None, continue_training: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Train model
        
        Args:
            nadata: nadata object
            verbose: Verbosity level
                0=only show progress bar
                1=show training loss, training regularization loss, validation loss, validation regularization loss
                2=on top of verbose=1, additionally show evaluation metrics like F1, AUC, Recall, Precision
            max_epochs: Maximum training epochs, if None use epochs from configuration
            continue_training: Whether to continue training (for tailor strategy)
            **kwargs: Additional parameters
            
        Returns:
            Training result dictionary
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        # Set CUDA debug environment variables
        if self.device.type == 'cuda':
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            self.logger.info("Enabled CUDA synchronous execution mode, helpful for debugging CUDA errors")
        
        # Prepare data
        X = nadata.X
        
        # Get labels
        config = nadata.Model.get_config()

        # Check if phenotype data exists
        if not hasattr(nadata, 'Meta') or nadata.Meta is None:
            raise ValueError(f"Phenotype data not found, please check if data loading is correct")

        # Get label data
        y = nadata.Meta["target"].values
        

        # Get existing data indices
        train_indices = nadata.Model.get_indices('train')
        test_indices = nadata.Model.get_indices('test')
        

        # Use existing train and test indices
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Get data corresponding to train indices
        X_train_full = X[train_indices]
        y_train_full = y[train_indices]

        # Further split train data into train and validation
        val_size = config.get('dataset', {}).get('val_size', 0.2)
        random_state = config.get('dataset', {}).get('random_state', 42)

        # Split validation from train data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full
        )

        # Calculate new train and validation indices
        n_train_full = len(train_indices)

        # Calculate validation position in original train indices
        val_size_adjusted = val_size
        n_val = int(n_train_full * val_size_adjusted)
        n_train = n_train_full - n_val

        # Update indices
        train_indices_final = train_indices[:n_train]
        val_indices = train_indices[n_train:]

        # Save updated indices
        nadata.Model.set_indices(
            train_idx=train_indices_final.tolist(),
            test_idx=test_indices.tolist(),
            val_idx=val_indices.tolist()
        )

        # Training parameters
        training_config = config.get('training', {})
        if max_epochs is None:
            epochs = training_config.get('epochs', 100)
        else:
            epochs = max_epochs
        learning_rate = training_config.get('learning_rate', 0.001)
        batch_size = training_config.get('batch_size', 32)
        reg_weight = training_config.get('regularization_weight', 0.1)
        
        # Convert to tensors and build TensorDataset
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)  # Changed to LongTensor
        
        # Build training dataset
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        
        # Add debug information
        self.logger.info(f"Training data shape: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
        self.logger.info(f"Training label value range: {y_train_tensor.min().item()} - {y_train_tensor.max().item()}")
        self.logger.info(f"Model output dimension: {self.model.output_dim}")
        
        # Build validation dataset (if validation data exists)
        val_dataset = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)  # Changed to LongTensor
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            self.logger.info(f"Validation data shape: X_val={X_val_tensor.shape}, y_val={y_val_tensor.shape}")
            self.logger.info(f"Validation label value range: {y_val_tensor.min().item()} - {y_val_tensor.max().item()}")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
        
        # Calculate class weights
        class_counts = torch.bincount(y_train_tensor)
        total_samples = y_train_tensor.size(0)
        num_classes = class_counts.size(0)
        
        # Ensure all classes have samples, avoid division by zero
        if num_classes > 0 and torch.min(class_counts) > 0:
            class_weights = total_samples / (num_classes * class_counts.float())
        else:
            # If some classes have no samples, use uniform weights
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        
        class_weights = class_weights.to(self.device)
        
        # Safety check: ensure label values do not exceed model output dimension
        max_label = y_train_tensor.max().item()
        model_output_dim = self.model.output_dim
        if max_label >= model_output_dim:
            raise ValueError(f"Label value ({max_label}) exceeds model output dimension ({model_output_dim}), please check data preprocessing")
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss(weight=class_weights)  # Changed to NLLLoss
        
        # Model initialization phase - train indicator of TrainableGeneSetLayer
        if not continue_training:
            self.logger.info("🔧 Starting model initialization phase - training geneset layer indicator...")
            
            # Decide whether to enable assist_layer in the initialization phase
            if self.model.use_assist_in_init:
                self.model.set_assist_layer_mode(True)
                self.logger.info("📊 Initialization phase: Enabled assist layer, directly map geneset output to probability")
            else:
                self.model.set_assist_layer_mode(False)
                self.logger.info("📊 Initialization phase: Using standard mode, use focus_layer for prediction")
            
            init_results = self._initialize_geneset_layer(train_loader, optimizer, verbose)
            self.logger.info(f"✅ Model initialization complete: {init_results}")
            
            # After initialization, switch to standard mode (using focus_layer)
            self.model.set_assist_layer_mode(False)
            self.logger.info("🔄 Initialization complete: Switching to standard mode, using focus_layer for prediction")
            
            # Save initialization results to nadata.uns
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            nadata.uns['init_results'] = init_results
            self.logger.info("💾 Initialization results saved to nadata.uns")
        else:
            # During continued training, ensure standard mode is used
            self.model.set_assist_layer_mode(False)
            self.logger.info("🔄 Continuing training: Using standard mode, using focus_layer for prediction")
        
        # Early stopping parameters
        patience = training_config.get('patience', 10)
        min_delta = 1e-6  # Minimum improvement threshold
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False
        
        # Add checkpoint saving related variables
        best_model_state = None
        best_epoch = 0
        outdir = config.get('global', {}).get('outdir', 'experiment/test')
        
        # Training loop
        train_losses = {'loss': [], 'reg_loss': []}
        val_losses = {'loss': [], 'reg_loss': []}
        
        if verbose >= 1:
            if continue_training:
                self.logger.info(f"Continuing training NNEA model... (remaining {epochs} epochs)")
            else:
                self.logger.info("Starting formal training NNEA model...")
            self.logger.info(f"Early stopping configuration: patience={patience}, min_delta={min_delta}")
            self.logger.info(f"Checkpoint saving directory: {outdir}")
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        # Create progress bar (only shown when verbose=0)
        if verbose == 0 and use_tqdm:
            pbar = tqdm(range(epochs), desc="Training progress")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # Training mode
            self.model.train()
            epoch_loss = 0.0
            epoch_reg_loss = 0.0
            num_batches = 0
            train_predictions = []
            train_targets = []
            
            # Use data loader for batch training
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # Move data to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    outputs = self.model(batch_X)
                    
                    # Check if outputs contain NaN or inf
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: Model outputs contain NaN or inf values")
                        continue
                    
                    loss = criterion(outputs, batch_y)
                    
                    # Check if loss value is valid
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss value is NaN or inf")
                        continue
                    
                    # Add regularization loss
                    reg_loss = self.model.regularization_loss()
                    total_loss = loss + reg_weight * reg_loss
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_reg_loss += reg_loss.item()
                    num_batches += 1
                    
                    # Collect training predictions for metric calculation
                    if verbose >= 2:
                        predictions = outputs.cpu().detach().numpy()
                        targets = batch_y.cpu().detach().numpy()
                        train_predictions.append(predictions)
                        train_targets.append(targets)
                    
                except Exception as e:
                    self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: Error during training: {e}")
                    continue
            
            # Calculate average loss
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_reg_loss = epoch_reg_loss / num_batches
                train_losses['loss'].append(avg_loss)
                train_losses['reg_loss'].append(avg_reg_loss)
                
                # Show training loss when verbose=1
                if verbose >= 1:
                    self.logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Reg_Loss={avg_reg_loss:.4f}")
                
                # Calculate and show training set evaluation metrics when verbose=2
                if verbose >= 2 and train_predictions and train_targets:
                    # Concatenate predictions from all batches
                    all_train_predictions = np.vstack(train_predictions)
                    all_train_targets = np.concatenate(train_targets)
                    
                    # Calculate training set evaluation metrics
                    train_metrics = self._calculate_validation_metrics(all_train_predictions, all_train_targets)
                    
                    # Show training set evaluation metrics
                    train_metrics_info = f"Epoch {epoch} Train Metrics: AUC={train_metrics['auc']:.4f}, F1={train_metrics['f1']:.4f}, Recall={train_metrics['recall']:.4f}, Precision={train_metrics['precision']:.4f}"
                    self.logger.info(train_metrics_info)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_reg_loss = 0.0
                val_num_batches = 0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        try:
                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y)
                            reg_loss = self.model.regularization_loss()
                            
                            val_loss += loss.item()
                            val_reg_loss += reg_loss.item()
                            val_num_batches += 1
                            
                            # Collect predictions for metric calculation
                            if verbose >= 2:
                                predictions = outputs.cpu().detach().numpy()
                                targets = batch_y.cpu().detach().numpy()
                                val_predictions.append(predictions)
                                val_targets.append(targets)
                            
                        except Exception as e:
                            self.logger.error(f"Error during validation: {e}")
                            continue
                
                if val_num_batches > 0:
                    avg_val_loss = val_loss / val_num_batches
                    avg_val_reg_loss = val_reg_loss / val_num_batches
                    val_losses['loss'].append(avg_val_loss)
                    val_losses['reg_loss'].append(avg_val_reg_loss)
                    
                    # Show validation loss when verbose=1
                    if verbose >= 1:
                        self.logger.info(f"Epoch {epoch} Validation: Val Loss={avg_val_loss:.4f}, Val Reg_Loss={avg_val_reg_loss:.4f}")
                    
                    # Calculate and show evaluation metrics when verbose=2
                    if verbose >= 2 and val_predictions and val_targets:
                        # Concatenate predictions from all batches
                        all_predictions = np.vstack(val_predictions)
                        all_targets = np.concatenate(val_targets)
                        
                        # Calculate evaluation metrics
                        val_metrics = self._calculate_validation_metrics(all_predictions, all_targets)
                        
                        # Show evaluation metrics
                        metrics_info = f"Epoch {epoch} Vak Metrics: AUC={val_metrics['auc']:.4f}, F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}, Precision={val_metrics['precision']:.4f}"
                        self.logger.info(metrics_info)
                
                # Early stopping check and checkpoint saving
                if val_loader is not None and avg_val_loss is not None:
                    # Check if validation loss improved
                    if avg_val_loss < best_val_loss - min_delta:
                        best_val_loss = avg_val_loss
                        best_epoch = epoch
                        patience_counter = 0
                        
                        # Save best model state
                        best_model_state = self.model.state_dict().copy()
                        
                        # Save checkpoint
                        checkpoint_path = os.path.join(outdir, f"checkpoint_epoch_{epoch}.pth")
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': best_model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': best_val_loss,
                                'train_loss': avg_loss if num_batches > 0 else None,
                                'train_reg_loss': avg_reg_loss if num_batches > 0 else None,
                                'val_reg_loss': avg_val_reg_loss
                            }, checkpoint_path)
                            if verbose >= 1:
                                self.logger.info(f"✅ Epoch {epoch}: Validation loss improved to {best_val_loss:.4f}")
                                self.logger.info(f"💾 Checkpoint saved to: {checkpoint_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to save checkpoint: {e}")
                    else:
                        patience_counter += 1
                        if verbose >= 1:
                            self.logger.info(f"⚠️ Epoch {epoch}: Validation loss did not improve, patience_counter={patience_counter}/{patience}")
                    
                    # Check if early stopping triggered
                    if patience_counter >= patience:
                        early_stopped = True
                        self.logger.info(f"🛑 Epoch {epoch}: Early stopping triggered! Validation loss did not improve for {patience} epochs")
                        self.logger.info(f"    Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
                        break
        
        # Training complete, restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"🔄 Best model restored (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")
            
            # Save final best model
            final_best_model_path = os.path.join(outdir, "best_model_final.pth")
            try:
                torch.save(best_model_state, final_best_model_path)
                self.logger.info(f"💾 Final best model saved to: {final_best_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to save final best model: {e}")
        
        # Training complete
        self.is_trained = True
        
        # Log early stopping information
        if early_stopped:
            self.logger.info(f"📊 Training ended due to early stopping, trained for {epoch+1} epochs")
        else:
            self.logger.info(f"📊 Training complete, trained for {epochs} epochs")
        
        # Return training results
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses['loss'][-1] if train_losses['loss'] else None,
            'final_val_loss': val_losses['loss'][-1] if val_losses['loss'] else None,
            'epochs_trained': epoch + 1 if early_stopped else epochs,
            'early_stopped': early_stopped,
            'best_val_loss': best_val_loss if val_loader is not None else None,
            'best_epoch': best_epoch if val_loader is not None else None,
            'patience_used': patience_counter if early_stopped else 0
        }
        
        # Include initialization results in return results
        if not continue_training and 'init_results' in locals():
            results['init_results'] = init_results
            # Also save to nadata.uns (if not already saved)
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            if 'init_results' not in nadata.uns:
                nadata.uns['init_results'] = init_results
        
        return results

    def _calculate_validation_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate validation metrics
        
        Args:
            predictions: Model predictions (N, num_classes)
            targets: True labels (N,)
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if predictions.shape[1] == 2:
                # Binary classification (softmax/log_softmax output, shape [N, 2])
                # Take positive class probability (assuming positive class is class 1)
                y_prob = predictions[:, 1]
                y_pred_binary = np.argmax(predictions, axis=1)
                auc = roc_auc_score(targets, y_prob)
                f1 = f1_score(targets, y_pred_binary)
                precision = precision_score(targets, y_pred_binary)
                recall = recall_score(targets, y_pred_binary)
            else:
                # Multi-class
                y_pred_binary = np.argmax(predictions, axis=1)
                auc = roc_auc_score(targets, predictions, multi_class='ovr')
                f1 = f1_score(targets, y_pred_binary, average='weighted')
                precision = precision_score(targets, y_pred_binary, average='weighted')
                recall = recall_score(targets, y_pred_binary, average='weighted')
            
            return {
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        except Exception as e:
            self.logger.error(f"Error calculating validation metrics: {e}")
            return {
                'auc': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

    def _initialize_geneset_layer(self, train_loader, optimizer, verbose: int = 1) -> Dict[str, Any]:
        """
        Initialize geneset layer - train indicator until condition is met
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            verbose: Verbosity level
            
        Returns:
            Dictionary of initialization results
        """
        self.logger.info("🔧 Starting geneset layer initialization...")
        
        # Confirm current assist_layer mode
        if self.model.get_assist_layer_mode():
            self.logger.info("📊 Initialization phase: Using assist layer, directly map geneset output to probability")
        else:
            self.logger.warning("⚠️ Initialization phase: Assist layer not used, it is recommended to enable assist_layer in the initialization phase")
        
        # Get geneset layer configuration
        config = self.config.get('nnea', {}).get('geneset_layer', {})
        geneset_threshold = config.get('geneset_threshold', 1e-5)
        max_set_size = config.get('max_set_size', 50)
        init_max_epochs = config.get('init_max_epochs', 100)
        init_patience = config.get('init_patience', 10)
        
        # Get initialization phase loss weight configuration
        init_task_loss_weight = config.get('init_task_loss_weight', 1.0)
        init_reg_loss_weight = config.get('init_reg_loss_weight', 10.0)
        init_total_loss_weight = config.get('init_total_loss_weight', 1.0)
        
        self.logger.info(f"Initialization parameters: geneset_threshold={geneset_threshold}, max_set_size={max_set_size}")
        self.logger.info(f"Initialization loss weights: task_loss_weight={init_task_loss_weight}, reg_loss_weight={init_reg_loss_weight}, total_loss_weight={init_total_loss_weight}")
        
        # Initialize variables
        best_condition_count = float('inf')
        patience_counter = 0
        init_epochs = 0
        
        # Initialize training loop
        for epoch in range(init_max_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Train one epoch
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    outputs = self.model(batch_X)
                    
                    # Calculate task loss (classification loss)
                    if self.assist_type == "classification":
                        task_loss = self._calculate_task_loss(outputs, batch_y)
                    elif self.assist_type == "rec":
                        task_loss = self._calculate_task_loss(outputs, batch_X)

                    # Calculate regularization loss
                    reg_loss = self.model.regularization_loss()
                    
                    # Calculate total loss (using configured weights)
                    total_loss = (init_task_loss_weight * task_loss + 
                                init_reg_loss_weight * reg_loss) * init_total_loss_weight
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += reg_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Initialization Epoch {epoch}, Batch: Error during training: {e}")
                    continue
            
            # Check geneset condition
            condition_met = self._check_geneset_condition(geneset_threshold, max_set_size)
            
            if condition_met:
                init_epochs = epoch + 1
                self.logger.info(f"✅ Geneset layer initialization complete, condition met in {init_epochs} epochs")
                break
            
            # Check if maximum epochs reached
            if epoch == init_max_epochs - 1:
                self.logger.warning(f"⚠️ Maximum initialization epochs ({init_max_epochs}) reached, forcing termination of initialization")
                init_epochs = init_max_epochs
                break
            
            # Early stopping check
            current_condition_count = self._count_genesets_above_threshold(geneset_threshold, max_set_size)
            total_gene_sets = self.model.geneset_layer.num_sets if hasattr(self.model, 'geneset_layer') else self.model.gene_set_layer.num_sets
            # Only start early stopping mechanism when current_condition_count starts decreasing (i.e., less than total_gene_sets)
            if current_condition_count < total_gene_sets:
                if current_condition_count < best_condition_count:
                    best_condition_count = current_condition_count
                    patience_counter = 0
                else:
                    patience_counter += 1
            if patience_counter >= init_patience:
                self.logger.info(f"⚠️ Initialization early stopping, no improvement for {init_patience} epochs")
                init_epochs = epoch + 1
                break
            
            if verbose >= 2 and (epoch % 20 == 0 or epoch == init_max_epochs - 1):
                condition_count = total_gene_sets - current_condition_count
                self.logger.info(f"Initialization Epoch {epoch}: Reg Loss={epoch_loss/num_batches:.4f}, number of genesets satisfying condition: {condition_count}/{total_gene_sets}")

        # Return initialization results
        init_results = {
            'init_epochs': init_epochs,
            'geneset_threshold': geneset_threshold,
            'max_set_size': max_set_size,
            'init_task_loss_weight': init_task_loss_weight,
            'init_reg_loss_weight': init_reg_loss_weight,
            'init_total_loss_weight': init_total_loss_weight,
            'final_condition_met': self._check_geneset_condition(geneset_threshold, max_set_size),
            'final_condition_count': self._count_genesets_above_threshold(geneset_threshold, max_set_size)
        }
        
        return init_results
    
    def _calculate_task_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate task loss (classification loss)
        
        Args:
            outputs: Model outputs
            targets: True labels
            
        Returns:
            Task loss
        """
        # Use cross-entropy loss
        if self.assist_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        elif self.assist_type == 'rec':
            criterion = nn.MSELoss()
        return criterion(outputs, targets)
    
    def _check_geneset_condition(self, geneset_threshold: float, max_set_size: int) -> bool:
        """
        Check if geneset condition is met
        
        Args:
            geneset_threshold: Geneset threshold
            max_set_size: Maximum geneset size
            
        Returns:
            Whether condition is met
        """
        try:
            # Get geneset layer indicator matrix
            if hasattr(self.model, 'geneset_layer'):
                indicators = self.model.geneset_layer.get_set_indicators()
            elif hasattr(self.model, 'gene_set_layer'):
                indicators = self.model.gene_set_layer.get_set_indicators()
            else:
                self.logger.warning("Geneset layer not found, assuming condition met")
                return True  # If no geneset layer, assume condition is met
            
            # Check each geneset
            for i in range(indicators.shape[0]):
                gene_assignments = indicators[i]
                selected_count = torch.sum(gene_assignments >= geneset_threshold).item()
                
                if selected_count >= max_set_size:
                    return False  # One geneset exceeds max size, condition not met
            
            return True  # All genesets meet condition
            
        except Exception as e:
            self.logger.error(f"Error checking geneset condition: {e}")
            return True  # Assume condition is met on error
    
    def _count_genesets_above_threshold(self, geneset_threshold: float, max_set_size: int) -> int:
        """
        Count number of genesets above threshold
        
        Args:
            geneset_threshold: Geneset threshold
            max_set_size: Maximum geneset size
            
        Returns:
            Number of genesets above threshold
        """
        try:
            # Get geneset layer indicator matrix
            if hasattr(self.model, 'geneset_layer'):
                indicators = self.model.geneset_layer.get_set_indicators()
            elif hasattr(self.model, 'gene_set_layer'):
                indicators = self.model.gene_set_layer.get_set_indicators()
            else:
                return 0
            
            count = 0
            for i in range(indicators.shape[0]):
                gene_assignments = indicators[i]
                selected_count = torch.sum(gene_assignments >= geneset_threshold).item()
                
                if selected_count >= max_set_size:
                    count += 1
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error calculating number of genesets: {e}")
            return 0
    
    def save_model(self, save_path: str) -> None:
        """
        Save model state
        
        Args:
            save_path: Save path
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        # Save model state dictionary
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'device': self.device,
            'is_trained': self.is_trained
        }, save_path)
        
        self.logger.info(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load model state
        
        Args:
            load_path: Load path
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Load model state dictionary
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update other attributes
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        if 'is_trained' in checkpoint:
            self.is_trained = checkpoint['is_trained']
        
        self.logger.info(f"Model loaded from {load_path}")
    
    def predict(self, nadata) -> np.ndarray:
        """
        Model prediction
        
        Args:
            nadata: nadata object
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            X = nadata.X
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()
    
    def evaluate(self, nadata, split='test') -> Dict[str, float]:
        """
        Model evaluation
        
        Args:
            nadata: nadata object
            split: Data split for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get data indices
        indices = nadata.Model.get_indices(split)
        if indices is None:
            raise ValueError(f"Indices for {split} split not found")
        
        # Get data based on indices
        X = nadata.X[indices] # Transpose to (num_samples, num_genes)
        
        # Get target column name
        config = nadata.Model.get_config()
        target_col = config.get('dataset', {}).get('target_column', 'target')
        y = nadata.Meta.iloc[indices][target_col].values
        
        # Predict for specific dataset
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Calculate metrics
        if predictions.shape[1] == 2:
            # Binary classification (softmax/log_softmax output, shape [N, 2])
            # Take positive class probability (assuming positive class is class 1)
            y_prob = predictions[:, 1]
            y_pred_binary = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(y, y_pred_binary)
            auc = roc_auc_score(y, y_prob)
            f1 = f1_score(y, y_pred_binary)
            precision = precision_score(y, y_pred_binary)
            recall = recall_score(y, y_pred_binary)
        else:
            # Multi-class
            y_pred_binary = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(y, y_pred_binary)
            auc = roc_auc_score(y, predictions, multi_class='ovr')
            f1 = f1_score(y, y_pred_binary, average='weighted')
            precision = precision_score(y, y_pred_binary, average='weighted')
            recall = recall_score(y, y_pred_binary, average='weighted')
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        # Save evaluation results to Model container
        eval_results = nadata.Model.get_metadata('evaluation_results') or {}
        eval_results[split] = results
        nadata.Model.add_metadata('evaluation_results', eval_results)
        
        self.logger.info(f"Model evaluation complete - {split} split:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def explain(self, nadata, method='importance') -> Dict[str, Any]:
        """
        Model explanation
        
        Args:
            nadata: nadata object
            method: Explanation method
            
        Returns:
            Dictionary of explanation results
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if method == 'importance':
            try:
                # Get geneset assignments
                geneset_assignments = self.model.get_geneset_assignments().detach().cpu().numpy()
                
                # Use DeepLIFT to calculate geneset importance
                geneset_importance = self._calculate_geneset_importance_with_deeplift(nadata)
                
                # Get attention weights (placeholder)
                attention_weights = self.model.get_attention_weights().detach().cpu().numpy()
                
                # Feature importance uses geneset importance as a substitute
                feature_importance = geneset_importance
                
                # Calculate gene importance (based on geneset assignments and importance)
                gene_importance = np.zeros(self.model.input_dim, dtype=np.float32)
                for i in range(self.model.num_genesets):
                    # Ensure dimension match: geneset_assignments[i] is a gene vector, geneset_importance[i] is a scalar
                    # gene_importance += geneset_assignments[i].astype(np.float32) * float(geneset_importance[i])
                    gene_importance += geneset_assignments[i].astype(np.float32) 
            except Exception as e:
                self.logger.warning(f"Gene importance calculation failed: {e}")
                # Use simplified method
                gene_importance = np.random.rand(self.model.input_dim)
                geneset_importance = np.random.rand(self.model.num_genesets)
                attention_weights = np.random.rand(self.model.num_genesets)
                feature_importance = geneset_importance
            
            # Sort and get top 20 important genes
            top_indices = np.argsort(gene_importance)[::-1][:20]
            top_genes = [nadata.Var.iloc[i]['Gene'] for i in top_indices]
            top_scores = gene_importance[top_indices]
            
            # Print top 20 genes
            self.logger.info(f"  - Top 20 important genes:")
            self.logger.info(f"    {'Rank':<4} {'Gene Name':<15} {'Importance Score':<12}")
            self.logger.info(f"    {'-'*4} {'-'*15} {'-'*12}")
            for i, (gene, score) in enumerate(zip(top_genes, top_scores)):
                self.logger.info(f"    {i+1:<4} {gene:<15} {score:<12.4f}")
            
            # Gene set refinement and annotation
            genesets_annotated = {}
            
            try:
                # Get gene name list
                gene_names = nadata.Var['Gene'].tolist()
                
                # Get configuration parameters
                nnea_config = self.config.get('nnea', {})
                geneset_config = nnea_config.get('geneset_layer', {})
                min_set_size = geneset_config.get('min_set_size', 10)
                max_set_size = geneset_config.get('max_set_size', 50)
                
                # Refine genesets
                from nnea.utils.enrichment import refine_genesets
                # Get geneset_threshold parameter from model
                geneset_threshold = self.model.geneset_layer.geneset_threshold
                genesets_refined = refine_genesets(
                    geneset_assignments=geneset_assignments,
                    geneset_importance=geneset_importance,
                    gene_names=gene_names,
                    min_set_size=min_set_size,
                    max_set_size=max_set_size,
                    geneset_threshold=geneset_threshold
                )
                
                # If explain_knowledge is configured, annotate
                explain_knowledge_path = nadata.uns.get('explain_knowledge_path')
                if explain_knowledge_path and genesets_refined:
                    from nnea.utils.enrichment import annotate_genesets
                    genesets_annotated = annotate_genesets(
                        genesets=genesets_refined,
                        gmt_path=explain_knowledge_path,
                        pvalueCutoff=0.05
                    )
                    
                    self.logger.info(f"Gene set annotation complete, annotated {len(genesets_annotated)} results")
                
            except Exception as e:
                self.logger.warning(f"Gene set refinement and annotation failed: {e}")
                # Use simplified geneset creation method
                if len(top_genes) >= 10:
                    genesets_refined = [
                        top_genes[:5],  # First 5 genes
                        top_genes[5:10]  # 6th to 10th genes
                    ]
            
            explain_results = {
                'importance': {
                    'top_genes': top_genes,
                    'importance_scores': top_scores.tolist(),
                    'genesets': genesets_annotated,
                    'geneset_importance': geneset_importance.tolist(),
                    'attention_weights': attention_weights.tolist(),
                    'feature_importance': feature_importance.tolist(),
                    'geneset_assignments': geneset_assignments.tolist()
                }
            }
            
            # Save explanation results
            nadata.uns['nnea_explain'] = explain_results
            
            self.logger.info(f"Model explanation complete:")

            # Output detailed information in descending order of geneset_importance
            self.logger.info(f"  - Geneset importance sorting results:")

            # Create sorting index
            sorted_indices = np.argsort(geneset_importance.flatten())[::-1]

            # Get gene name list
            gene_names = nadata.Var['Gene'].tolist()

            # Output header
            self.logger.info(f"    {'Importance Score':<12} {'Geneset Key':<30} {'Top Genes':<50}")
            self.logger.info(f"    {'-'*12} {'-'*30} {'-'*50}")
            
            # Output in descending order of importance
            for i, idx in enumerate(sorted_indices):
                if i >= 20:  # Only show top 20
                    remaining = len(sorted_indices) - 20
                    if remaining > 0:
                        self.logger.info(f"    ... and {remaining} more genesets")
                    break
                
                importance_score = geneset_importance.flatten()[idx]
                
                # Get corresponding geneset key
                geneset_key = f"Geneset_{idx}"
                if genesets_annotated and idx < len(genesets_annotated):
                    # Get keys from genesets_annotated
                    keys_list = list(genesets_annotated.keys())
                    if idx < len(keys_list):
                        geneset_key = keys_list[idx]
                
                # Get top genes assigned to this geneset
                # Based on geneset_assignments matrix, find important genes assigned to this geneset
                gene_assignments = geneset_assignments[idx]  # Gene assignments weights for this geneset
                top_gene_indices = np.argsort(gene_assignments)[::-1][:5]  # Take top 5 most important genes
                top_genes_for_geneset = [gene_names[j] for j in top_gene_indices if j < len(gene_names)]
                top_genes_str = ", ".join(top_genes_for_geneset)
                
                self.logger.info(f"    {importance_score:<12.4f} {geneset_key:<30} {top_genes_str:<50}")
            
            return explain_results
        else:
            raise ValueError(f"Unsupported explanation method: {method}")
    
    def _calculate_geneset_importance_with_deeplift(self, nadata) -> np.ndarray:
        """
        Calculate geneset importance using DeepLIFT
        
        Args:
            nadata: nadata object
            
        Returns:
            Array of geneset importance
        """
        self.model.eval()
        
        # Get data
        X = nadata.X
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Prepare input for geneset layer
        R, S = self.model._prepare_input_for_geneset(X_tensor)
        
        # Calculate integrated gradients for all samples
        all_ig_scores = []
        
        for i in range(min(100, len(X))):  # Limit sample number for efficiency
            # Get single sample
            R_sample = R[i:i+1]
            S_sample = S[i:i+1]
            
            # Calculate integrated gradients for this sample
            ig_score = self._integrated_gradients_for_genesets(
                R_sample, S_sample, steps=50
            )
            all_ig_scores.append(ig_score.cpu().numpy())
        
        # Calculate average importance scores
        avg_ig_scores = np.mean(all_ig_scores, axis=0)
        
        return avg_ig_scores
    
    def _integrated_gradients_for_genesets(self, R, S, target_class=None, baseline=None, steps=50):
        """
        Explain geneset importance using integrated gradients
        
        Args:
            R: Gene expression data (1, num_genes)
            S: Gene sorting index (1, num_genes)
            target_class: Target class to explain (default is model's predicted class)
            baseline: Baseline value for geneset (default is all zeros vector)
            steps: Number of interpolation steps
            
        Returns:
            ig: Importance score for genesets (num_sets,)
        """
        # Ensure input is single sample
        assert R.shape[0] == 1 and S.shape[0] == 1, "Single sample explanation only supported"
        
        # Calculate enrichment scores (es_scores)
        with torch.no_grad():
            es_scores = self.model.geneset_layer(R, S)  # (1, num_sets)
        
        # Determine target class
        if target_class is None:
            with torch.no_grad():
                # Reconstruct original input x from R and S
                x = R  # For models in NNEA package, R is the original input
                output = self.model(x)
                if self.model.output_dim == 1:
                    target_class = 0  # Binary classification
                else:
                    target_class = torch.argmax(output, dim=1).item()
        
        # Set baseline value
        if baseline is None:
            baseline = torch.zeros_like(es_scores)
        
        # Generate interpolation path (steps points)
        scaled_es_scores = []
        for step in range(1, steps + 1):
            alpha = step / steps
            interpolated = baseline + alpha * (es_scores - baseline)
            scaled_es_scores.append(interpolated)
        
        # Store gradients
        gradients = []
        
        # Calculate gradients at interpolation points
        for interp_es in scaled_es_scores:
            interp_es = interp_es.clone().requires_grad_(True)
            
            # Handle output dimension
            if self.model.output_dim == 1:
                # Binary classification
                logits = self.model.focus_layer(interp_es)
                target_logit = logits.squeeze()
            else:
                # Multi-class
                logits = self.model.focus_layer(interp_es)
                target_logit = logits[0, target_class]
            
            # Calculate gradient
            grad = torch.autograd.grad(outputs=target_logit, inputs=interp_es)[0]
            gradients.append(grad.detach())
        
        # Integrate gradients to calculate integrated gradients
        gradients = torch.stack(gradients)  # (steps, 1, num_sets)
        avg_gradients = torch.mean(gradients, dim=0)  # (1, num_sets)
        ig = (es_scores - baseline) * avg_gradients  # (1, num_sets)
        
        return ig.squeeze(0)  # (num_sets,) 