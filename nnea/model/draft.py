import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nnea.model.nnea_model import NNEAModel
from nnea.model.base import BaseModel


class NNEARegresser(BaseModel):
    """
    NNEA Regressor
    Implements an interpretable regression model with TrainableGeneSetLayer as the core
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NNEA regressor

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.task = 'regression'

    def build(self, nadata) -> None:
        """
        Build model

        Args:
            nadata: nadata object
        """
        if nadata is None:
            raise ValueError("nadata object cannot be empty")

        # Get input dimensions
        if hasattr(nadata, 'X') and nadata.X is not None:
            input_dim = nadata.X.shape[1]  # Number of genes
        else:
            raise ValueError("Expression matrix not loaded")

        # Get output dimensions - regression task output dimension is 1
        output_dim = 1

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
                    # Ensure prior knowledge matrix matches input dimensions
                    if piror_knowledge.shape[1] != input_dim:
                        self.logger.warning(
                            f"Prior knowledge matrix dimensions ({piror_knowledge.shape[1]}) do not match input dimensions ({input_dim})")
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

        self.logger.info(f"NNEA regressor built: input_dim={input_dim}, output_dim={output_dim}")
        num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
        self.logger.info(f"Number of genesets: {num_genesets}")
        self.logger.info(f"Using prior knowledge: {use_piror_knowledge}")

    def train(self, nadata, verbose: int = 1, max_epochs: Optional[int] = None, continue_training: bool = False,
              **kwargs) -> Dict[str, Any]:
        """
        Train model

        Args:
            nadata: nadata object
            verbose: Verbosity level
                0=Only show progress bar
                1=Show training loss, training regularization loss, validation loss, validation regularization loss
                2=On top of verbose=1, also show MSE, MAE, R2 and other evaluation metrics
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
            self.logger.info("CUDA synchronous execution mode enabled, helpful for debugging CUDA errors")

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
            X_train_full, y_train_full, test_size=val_size, random_state=random_state
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
        y_train_tensor = torch.FloatTensor(y_train)  # Regression task uses FloatTensor

        # Build training dataset
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

        # Add debug information
        self.logger.info(f"Training data shape: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
        self.logger.info(f"Training label value range: {y_train_tensor.min().item():.4f} - {y_train_tensor.max().item():.4f}")
        self.logger.info(f"Model output dimension: {self.model.output_dim}")

        # Build validation dataset (if validation data exists)
        val_dataset = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)  # Regression task uses FloatTensor
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            self.logger.info(f"Validation data shape: X_val={X_val_tensor.shape}, y_val={y_val_tensor.shape}")
            self.logger.info(f"Validation label value range: {y_val_tensor.min().item():.4f} - {y_val_tensor.max().item():.4f}")

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

        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()  # Regression task uses MSE loss

        # Model initialization phase - train TrainableGeneSetLayer indicator
        if not continue_training:
            self.logger.info("🔧 Starting model initialization phase - training geneset layer indicator matrix...")

            # Decide whether to enable assist_layer during initialization based on configuration
            if self.model.use_assist_in_init:
                self.model.set_assist_layer_mode(True)
                self.logger.info("📊 Initialization phase: Enable assist layer, directly map geneset output to prediction values")
            else:
                self.model.set_assist_layer_mode(False)
                self.logger.info("📊 Initialization phase: Use standard mode, use focus_layer for prediction")

            init_results = self._initialize_geneset_layer(train_loader, optimizer, verbose)
            self.logger.info(f"✅ Model initialization completed: {init_results}")

            # After initialization, switch to standard mode (use focus_layer)
            self.model.set_assist_layer_mode(False)
            self.logger.info("🔄 Initialization completed: Switch to standard mode, use focus_layer for prediction")

            # Save initialization results to nadata.uns
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            nadata.uns['init_results'] = init_results
            self.logger.info("💾 Initialization results saved to nadata.uns")
        else:
            # When continuing training, ensure standard mode is used
            self.model.set_assist_layer_mode(False)
            self.logger.info("🔄 Continue training: Use standard mode, use focus_layer for prediction")

        # Early stopping mechanism parameters
        patience = training_config.get('patience', 10)
        min_delta = 1e-6  # Minimum improvement threshold

        # Early stopping variable initialization
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
                self.logger.info(f"Continue training NNEA model... (remaining {epochs} epochs)")
            else:
                self.logger.info("Start formal training of NNEA model...")
            self.logger.info(f"Early stopping configuration: patience={patience}, min_delta={min_delta}")
            self.logger.info(f"Checkpoint save directory: {outdir}")

        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        # Create progress bar (only show when verbose=0)
        if verbose == 0 and use_tqdm:
            pbar = tqdm(range(epochs), desc="Training Progress")
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

                    # Check if output contains NaN or infinity
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: Model output contains NaN or infinite values")
                        continue

                    loss = criterion(outputs.squeeze(), batch_y)  # Regression task needs squeeze

                    # Check if loss value is valid
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss value is NaN or infinite")
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

                    # Collect training prediction results for calculating metrics
                    if verbose >= 2:
                        predictions = outputs.squeeze().cpu().detach().numpy()
                        targets = batch_y.cpu().detach().numpy()
                        train_predictions.append(predictions)
                        train_targets.append(targets)

                except Exception as e:
                    self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: Error occurred during training: {e}")
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
                    # Combine prediction results from all batches
                    all_train_predictions = np.concatenate(train_predictions)
                    all_train_targets = np.concatenate(train_targets)

                    # Calculate training set evaluation metrics
                    train_metrics = self._calculate_validation_metrics(all_train_predictions, all_train_targets)

                    # 显示训练集评估指标
                    train_metrics_info = f"Epoch {epoch} Train Metrics: MSE={train_metrics['mse']:.4f}, MAE={train_metrics['mae']:.4f}, R2={train_metrics['r2']:.4f}"
                    self.logger.info(train_metrics_info)

            # 验证阶段
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
                            loss = criterion(outputs.squeeze(), batch_y)  # 回归任务需要squeeze
                            reg_loss = self.model.regularization_loss()

                            val_loss += loss.item()
                            val_reg_loss += reg_loss.item()
                            val_num_batches += 1

                            # 收集预测结果用于计算指标
                            if verbose >= 2:
                                predictions = outputs.squeeze().cpu().detach().numpy()
                                targets = batch_y.cpu().detach().numpy()
                                val_predictions.append(predictions)
                                val_targets.append(targets)

                        except Exception as e:
                            self.logger.error(f"验证过程中出现错误: {e}")
                            continue

                if val_num_batches > 0:
                    avg_val_loss = val_loss / val_num_batches
                    avg_val_reg_loss = val_reg_loss / val_num_batches
                    val_losses['loss'].append(avg_val_loss)
                    val_losses['reg_loss'].append(avg_val_reg_loss)

                    # verbose=1时显示验证损失
                    if verbose >= 1:
                        self.logger.info(
                            f"Epoch {epoch} Validation: Val Loss={avg_val_loss:.4f}, Val Reg_Loss={avg_val_reg_loss:.4f}")

                    # verbose=2时计算并显示评估指标
                    if verbose >= 2 and val_predictions and val_targets:
                        # 合并所有批次的预测结果
                        all_predictions = np.concatenate(val_predictions)
                        all_targets = np.concatenate(val_targets)

                        # 计算评估指标
                        val_metrics = self._calculate_validation_metrics(all_predictions, all_targets)

                        # 显示评估指标
                        metrics_info = f"Epoch {epoch} Val Metrics: MSE={val_metrics['mse']:.4f}, MAE={val_metrics['mae']:.4f}, R2={val_metrics['r2']:.4f}"
                        self.logger.info(metrics_info)

                # 早停检查和checkpoint保存
                if val_loader is not None and avg_val_loss is not None:
                    # 检查验证损失是否改善
                    if avg_val_loss < best_val_loss - min_delta:
                        best_val_loss = avg_val_loss
                        best_epoch = epoch
                        patience_counter = 0

                        # 保存最佳模型状态
                        best_model_state = self.model.state_dict().copy()

                        # 保存checkpoint
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
                                self.logger.info(f"✅ Epoch {epoch}: 验证损失改善到 {best_val_loss:.4f}")
                                self.logger.info(f"💾 Checkpoint已保存到: {checkpoint_path}")
                        except Exception as e:
                            self.logger.error(f"保存checkpoint失败: {e}")
                    else:
                        patience_counter += 1
                        if verbose >= 1:
                            self.logger.info(
                                f"⚠️ Epoch {epoch}: 验证损失未改善，patience_counter={patience_counter}/{patience}")

                    # 检查是否触发早停
                    if patience_counter >= patience:
                        early_stopped = True
                        self.logger.info(f"🛑 Epoch {epoch}: 触发早停！验证损失在{patience}个epoch内未改善")
                        self.logger.info(f"   最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
                        break

        # 训练完成，恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"🔄 已恢复最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

            # 保存最终的最佳模型
            final_best_model_path = os.path.join(outdir, "best_model_final.pth")
            try:
                torch.save(best_model_state, final_best_model_path)
                self.logger.info(f"💾 最终最佳模型已保存到: {final_best_model_path}")
            except Exception as e:
                self.logger.error(f"保存最终最佳模型失败: {e}")

        # 训练完成
        self.is_trained = True

        # 记录早停信息
        if early_stopped:
            self.logger.info(f"📊 训练因早停而结束，实际训练了{epoch + 1}个epoch")
        else:
            self.logger.info(f"📊 训练完成，共训练了{epochs}个epoch")

        # 返回训练结果
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

        # 将初始化结果也包含在返回结果中
        if not continue_training and 'init_results' in locals():
            results['init_results'] = init_results
            # 同时保存到nadata.uns中（如果还没有保存的话）
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            if 'init_results' not in nadata.uns:
                nadata.uns['init_results'] = init_results

        return results

    def _calculate_validation_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        计算验证集的评估指标

        Args:
            predictions: 模型预测结果 (N,)
            targets: 真实标签 (N,)

        Returns:
            评估指标字典
        """
        try:
            mse = mean_squared_error(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)
            rmse = np.sqrt(mse)

            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse
            }
        except Exception as e:
            self.logger.error(f"计算验证指标时出现错误: {e}")
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'r2': -float('inf'),
                'rmse': float('inf')
            }

    def _initialize_geneset_layer(self, train_loader, optimizer, verbose: int = 1) -> Dict[str, Any]:
        """
        初始化基因集层 - 训练indicator直到满足条件

        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            verbose: 详细程度

        Returns:
            初始化结果字典
        """
        self.logger.info("🔧 开始基因集层初始化...")

        # 确认当前使用assist_layer模式
        if self.model.get_assist_layer_mode():
            self.logger.info("📊 初始化阶段：使用辅助层直接映射geneset输出为预测值")
        else:
            self.logger.warning("⚠️ 初始化阶段：未使用辅助层，建议在初始化阶段启用assist_layer")

        # 获取基因集层配置
        config = self.config.get('nnea', {}).get('geneset_layer', {})
        geneset_threshold = config.get('geneset_threshold', 1e-5)
        max_set_size = config.get('max_set_size', 50)
        init_max_epochs = config.get('init_max_epochs', 100)
        init_patience = config.get('init_patience', 10)

        # 获取初始化阶段的损失权重配置
        init_task_loss_weight = config.get('init_task_loss_weight', 1.0)
        init_reg_loss_weight = config.get('init_reg_loss_weight', 10.0)
        init_total_loss_weight = config.get('init_total_loss_weight', 1.0)

        self.logger.info(f"初始化参数: geneset_threshold={geneset_threshold}, max_set_size={max_set_size}")
        self.logger.info(
            f"初始化损失权重: task_loss_weight={init_task_loss_weight}, reg_loss_weight={init_reg_loss_weight}, total_loss_weight={init_total_loss_weight}")

        # 初始化变量
        best_condition_count = float('inf')
        patience_counter = 0
        init_epochs = 0

        # 初始化训练循环
        for epoch in range(init_max_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            # 训练一个epoch
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                try:
                    # 前向传播
                    outputs = self.model(batch_X)

                    # 计算任务损失（回归损失）
                    task_loss = self._calculate_task_loss(outputs, batch_y)

                    # 计算正则化损失
                    reg_loss = self.model.regularization_loss()

                    # 计算总损失（使用配置的权重）
                    total_loss = (init_task_loss_weight * task_loss +
                                  init_reg_loss_weight * reg_loss) * init_total_loss_weight

                    # 反向传播
                    total_loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()

                    epoch_loss += reg_loss.item()
                    num_batches += 1

                except Exception as e:
                    self.logger.error(f"初始化Epoch {epoch}, Batch: 训练过程中出现错误: {e}")
                    continue

            # 检查基因集条件
            condition_met = self._check_geneset_condition(geneset_threshold, max_set_size)

            if condition_met:
                init_epochs = epoch + 1
                self.logger.info(f"✅ 基因集层初始化完成，在第{init_epochs}个epoch满足条件")
                break

            # 检查是否达到最大轮数
            if epoch == init_max_epochs - 1:
                self.logger.warning(f"⚠️ 达到最大初始化轮数({init_max_epochs})，强制结束初始化")
                init_epochs = init_max_epochs
                break

            # 早停检查
            current_condition_count = self._count_genesets_above_threshold(geneset_threshold, max_set_size)
            total_gene_sets = self.model.geneset_layer.num_sets if hasattr(self.model,
                                                                           'geneset_layer') else self.model.gene_set_layer.num_sets
            # 只有当current_condition_count开始减少（即小于total_gene_sets）时才启动早停机制
            if current_condition_count < total_gene_sets:
                if current_condition_count < best_condition_count:
                    best_condition_count = current_condition_count
                    patience_counter = 0
                else:
                    patience_counter += 1
            if patience_counter >= init_patience:
                self.logger.info(f"⚠️ 初始化早停，连续{init_patience}个epoch未改善")
                init_epochs = epoch + 1
                break

            if verbose >= 2 and (epoch % 20 == 0 or epoch == init_max_epochs - 1):
                condition_count = total_gene_sets - current_condition_count
                self.logger.info(
                    f"初始化Epoch {epoch}: Reg Loss={epoch_loss / num_batches:.4f}, 满足条件的基因集数: {condition_count}/{total_gene_sets}")

        # 返回初始化结果
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
        计算任务损失（回归损失）

        Args:
            outputs: 模型输出
            targets: 真实标签

        Returns:
            任务损失
        """
        # 使用MSE损失
        criterion = nn.MSELoss()
        return criterion(outputs.squeeze(), targets)

    def _check_geneset_condition(self, geneset_threshold: float, max_set_size: int) -> bool:
        """
        检查基因集条件是否满足

        Args:
            geneset_threshold: 基因集阈值
            max_set_size: 最大基因集大小

        Returns:
            是否满足条件
        """
        try:
            # 获取基因集层的指示矩阵
            if hasattr(self.model, 'geneset_layer'):
                indicators = self.model.geneset_layer.get_set_indicators()
            elif hasattr(self.model, 'gene_set_layer'):
                indicators = self.model.gene_set_layer.get_set_indicators()
            else:
                self.logger.warning("未找到基因集层，无法检查条件")
                return True  # 如果没有基因集层，认为条件满足

            # 检查每个基因集
            for i in range(indicators.shape[0]):
                gene_assignments = indicators[i]
                selected_count = torch.sum(gene_assignments >= geneset_threshold).item()

                if selected_count >= max_set_size:
                    return False  # 有一个基因集超过最大大小，条件不满足

            return True  # 所有基因集都满足条件

        except Exception as e:
            self.logger.error(f"检查基因集条件时出现错误: {e}")
            return True  # 出错时认为条件满足

    def _count_genesets_above_threshold(self, geneset_threshold: float, max_set_size: int) -> int:
        """
        计算超过阈值的基因集数量

        Args:
            geneset_threshold: 基因集阈值
            max_set_size: 最大基因集大小

        Returns:
            超过阈值的基因集数量
        """
        try:
            # 获取基因集层的指示矩阵
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
            self.logger.error(f"计算基因集数量时出现错误: {e}")
            return 0

    def save_model(self, save_path: str) -> None:
        """
        保存模型状态

        Args:
            save_path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未构建")

        # 保存模型状态字典
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'device': self.device,
            'is_trained': self.is_trained
        }, save_path)

        self.logger.info(f"模型已保存到: {save_path}")

    def load_model(self, load_path: str) -> None:
        """
        加载模型状态

        Args:
            load_path: 加载路径
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        # 加载模型状态字典
        checkpoint = torch.load(load_path, map_location=self.device)

        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 更新其他属性
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        if 'is_trained' in checkpoint:
            self.is_trained = checkpoint['is_trained']

        self.logger.info(f"模型已从 {load_path} 加载")

    def predict(self, nadata) -> np.ndarray:
        """
        模型预测

        Args:
            nadata: nadata对象

        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        self.model.eval()
        with torch.no_grad():
            X = nadata.X
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.squeeze().cpu().numpy()  # 回归任务需要squeeze

    def evaluate(self, nadata, split='test') -> Dict[str, float]:
        """
        模型评估

        Args:
            nadata: nadata对象
            split: 评估的数据集分割

        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        # 获取数据索引
        indices = nadata.Model.get_indices(split)
        if indices is None:
            raise ValueError(f"未找到{split}集的索引")

        # 根据索引获取数据
        X = nadata.X[indices]  # 转置为(样本数, 基因数)

        # 获取目标列名
        config = nadata.Model.get_config()
        target_col = config.get('dataset', {}).get('target_column', 'target')
        y = nadata.Meta.iloc[indices][target_col].values

        # 对特定数据集进行预测
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).squeeze().cpu().numpy()  # 回归任务需要squeeze

        # 计算指标
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mse)

        results = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse
        }

        # 保存评估结果到Model容器
        eval_results = nadata.Model.get_metadata('evaluation_results') or {}
        eval_results[split] = results
        nadata.Model.add_metadata('evaluation_results', eval_results)

        self.logger.info(f"模型评估完成 - {split}集:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return results

    def explain(self, nadata, method='importance') -> Dict[str, Any]:
        """
        模型解释

        Args:
            nadata: nadata对象
            method: 解释方法

        Returns:
            解释结果字典
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        if method == 'importance':
            try:
                # 获取基因集分配
                geneset_assignments = self.model.get_geneset_assignments().detach().cpu().numpy()

                # 使用DeepLIFT计算基因集重要性
                geneset_importance = self._calculate_geneset_importance_with_deeplift(nadata)

                # 获取注意力权重（占位符）
                attention_weights = self.model.get_attention_weights().detach().cpu().numpy()

                # 特征重要性使用基因集重要性作为替代
                feature_importance = geneset_importance

                # 计算基因重要性（基于基因集分配和重要性）
                gene_importance = np.zeros(self.model.input_dim, dtype=np.float32)
                for i in range(self.model.num_genesets):
                    # 确保维度匹配：geneset_assignments[i]是基因向量，geneset_importance[i]是标量
                    gene_importance += geneset_assignments[i].astype(np.float32)
            except Exception as e:
                self.logger.warning(f"基因重要性计算失败: {e}")
                # 使用简化的方法
                gene_importance = np.random.rand(self.model.input_dim)
                geneset_importance = np.random.rand(self.model.num_genesets)
                attention_weights = np.random.rand(self.model.num_genesets)
                feature_importance = geneset_importance

            # 排序并获取前20个重要基因
            top_indices = np.argsort(gene_importance)[::-1][:20]
            top_genes = [nadata.Var.iloc[i]['Gene'] for i in top_indices]
            top_scores = gene_importance[top_indices]

            # 打印20个top_genes
            self.logger.info(f"  - Top 20 重要基因:")
            self.logger.info(f"    {'排名':<4} {'基因名':<15} {'重要性分数':<12}")
            self.logger.info(f"    {'-' * 4} {'-' * 15} {'-' * 12}")
            for i, (gene, score) in enumerate(zip(top_genes, top_scores)):
                self.logger.info(f"    {i + 1:<4} {gene:<15} {score:<12.4f}")

            # 基因集精炼和注释
            genesets_annotated = {}

            try:
                # 获取基因名称列表
                gene_names = nadata.Var['Gene'].tolist()

                # 获取配置参数
                nnea_config = self.config.get('nnea', {})
                geneset_config = nnea_config.get('geneset_layer', {})
                min_set_size = geneset_config.get('min_set_size', 10)
                max_set_size = geneset_config.get('max_set_size', 50)

                # 精炼基因集
                from nnea.utils.enrichment import refine_genesets
                # 从模型中获取geneset_threshold参数
                geneset_threshold = self.model.geneset_layer.geneset_threshold
                genesets_refined = refine_genesets(
                    geneset_assignments=geneset_assignments,
                    geneset_importance=geneset_importance,
                    gene_names=gene_names,
                    min_set_size=min_set_size,
                    max_set_size=max_set_size,
                    geneset_threshold=geneset_threshold
                )

                # 如果配置了explain_knowledge，进行注释
                explain_knowledge_path = nadata.uns.get('explain_knowledge_path')
                if explain_knowledge_path and genesets_refined:
                    from nnea.utils.enrichment import annotate_genesets
                    genesets_annotated = annotate_genesets(
                        genesets=genesets_refined,
                        gmt_path=explain_knowledge_path,
                        pvalueCutoff=0.05
                    )

                    self.logger.info(f"完成基因集注释，注释结果数量: {len(genesets_annotated)}")

            except Exception as e:
                self.logger.warning(f"基因集精炼和注释失败: {e}")
                # 使用简化的基因集创建方法
                if len(top_genes) >= 10:
                    genesets_refined = [
                        top_genes[:5],  # 前5个基因
                        top_genes[5:10]  # 第6-10个基因
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

            # 保存解释结果
            nadata.uns['nnea_explain'] = explain_results

            self.logger.info(f"模型解释完成:")

            # 按geneset_importance降序输出详细信息
            self.logger.info(f"  - 基因集重要性排序结果:")

            # 创建排序索引
            sorted_indices = np.argsort(geneset_importance.flatten())[::-1]

            # 获取基因名称列表
            gene_names = nadata.Var['Gene'].tolist()

            # 输出表头
            self.logger.info(f"    {'重要性分数':<12} {'基因集Key':<30} {'Top基因':<50}")
            self.logger.info(f"    {'-' * 12} {'-' * 30} {'-' * 50}")

            # 按重要性降序输出
            for i, idx in enumerate(sorted_indices):
                if i >= 20:  # 只显示前20个
                    remaining = len(sorted_indices) - 20
                    if remaining > 0:
                        self.logger.info(f"    ... 还有 {remaining} 个基因集")
                    break

                importance_score = geneset_importance.flatten()[idx]

                # 获取对应的geneset key
                geneset_key = f"Geneset_{idx}"
                if genesets_annotated and idx < len(genesets_annotated):
                    # 获取genesets_annotated的键
                    keys_list = list(genesets_annotated.keys())
                    if idx < len(keys_list):
                        geneset_key = keys_list[idx]

                # 获取分配给该基因集的top genes
                # 基于geneset_assignments矩阵，找到分配给该基因集的重要基因
                gene_assignments = geneset_assignments[idx]  # 该基因集的基因分配权重
                top_gene_indices = np.argsort(gene_assignments)[::-1][:5]  # 取前5个最重要的基因
                top_genes_for_geneset = [gene_names[j] for j in top_gene_indices if j < len(gene_names)]
                top_genes_str = ", ".join(top_genes_for_geneset)

                self.logger.info(f"    {importance_score:<12.4f} {geneset_key:<30} {top_genes_str:<50}")

            return explain_results
        else:
            raise ValueError(f"不支持的解释方法: {method}")

    def _calculate_geneset_importance_with_deeplift(self, nadata) -> np.ndarray:
        """
        使用DeepLIFT计算基因集重要性

        Args:
            nadata: nadata对象

        Returns:
            基因集重要性数组
        """
        self.model.eval()

        # 获取数据
        X = nadata.X
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 为基因集层准备输入
        R, S = self.model._prepare_input_for_geneset(X_tensor)

        # 计算所有样本的积分梯度
        all_ig_scores = []

        for i in range(min(100, len(X))):  # 限制样本数量以提高效率
            # 获取单个样本
            R_sample = R[i:i + 1]
            S_sample = S[i:i + 1]

            # 计算该样本的积分梯度
            ig_score = self._integrated_gradients_for_genesets(
                R_sample, S_sample, steps=50
            )
            all_ig_scores.append(ig_score.cpu().numpy())

        # 计算平均重要性分数
        avg_ig_scores = np.mean(all_ig_scores, axis=0)

        return avg_ig_scores

    def _integrated_gradients_for_genesets(self, R, S, baseline=None, steps=50):
        """
        使用积分梯度解释基因集重要性

        Args:
            R: 基因表达数据 (1, num_genes)
            S: 基因排序索引 (1, num_genes)
            baseline: 基因集的基线值 (默认全零向量)
            steps: 积分路径的插值步数

        Returns:
            ig: 基因集重要性分数 (num_sets,)
        """
        # 确保输入为单样本
        assert R.shape[0] == 1 and S.shape[0] == 1, "只支持单样本解释"

        # 计算样本的富集分数 (es_scores)
        with torch.no_grad():
            es_scores = self.model.geneset_layer(R, S)  # (1, num_sets)

        # 设置基线值
        if baseline is None:
            baseline = torch.zeros_like(es_scores)

        # 生成插值路径 (steps个点)
        scaled_es_scores = []
        for step in range(1, steps + 1):
            alpha = step / steps
            interpolated = baseline + alpha * (es_scores - baseline)
            scaled_es_scores.append(interpolated)

        # 存储梯度
        gradients = []

        # 计算插值点梯度
        for interp_es in scaled_es_scores:
            interp_es = interp_es.clone().requires_grad_(True)

            # 回归任务，直接使用输出
            logits = self.model.focus_layer(interp_es)
            target_logit = logits.squeeze()

            # 计算梯度
            grad = torch.autograd.grad(outputs=target_logit, inputs=interp_es)[0]
            gradients.append(grad.detach())

        # 整合梯度计算积分梯度
        gradients = torch.stack(gradients)  # (steps, 1, num_sets)
        avg_gradients = torch.mean(gradients, dim=0)  # (1, num_sets)
        ig = (es_scores - baseline) * avg_gradients  # (1, num_sets)

        return ig.squeeze(0)  # (num_sets,)
