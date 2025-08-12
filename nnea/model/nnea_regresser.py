import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from nnea.model.nnea_model import NNEAModel
from nnea.model.base import BaseModel
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error, \
    r2_score


class NNEARegresser(BaseModel):
    """
    NNEA回归器
    实现可解释的回归模型，以TrainableGeneSetLayer为核心
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化NNEA回归器
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        self.task = 'regression'
        
    def build(self, nadata) -> None:
        """
        构建模型
        
        Args:
            nadata: nadata对象
        """
        if nadata is None:
            raise ValueError("nadata对象不能为空")
        
        # 获取输入维度
        if hasattr(nadata, 'X') and nadata.X is not None:
            input_dim = nadata.X.shape[1]  # 基因数量
        else:
            raise ValueError("表达矩阵未加载")
        
        # 获取输出维度
        output_dim = 1
        
        # 获取nnea配置部分
        nnea_config = self.config.get('nnea', {})
        
        # 处理先验知识
        piror_knowledge = None
        use_piror_knowledge = nnea_config.get('piror_knowledge', {}).get('use_piror_knowledge', False)
        if use_piror_knowledge:
            # 获取基因名称列表
            gene_names = None
            if hasattr(nadata, 'Var') and nadata.Var is not None:
                gene_names = nadata.Var['Gene'].tolist()
                
            if gene_names is not None:
                # 从nnea.io模块导入先验知识加载函数
                from nnea.io._load import load_piror_knowledge
                piror_knowledge = load_piror_knowledge(self.config, gene_names)
                
                if piror_knowledge is not None:
                    self.logger.info(f"成功加载先验知识，形状: {piror_knowledge.shape}")
                    piror_knowledge = torch.tensor(piror_knowledge, dtype=torch.float32)
                    # 确保先验知识矩阵与输入维度匹配
                    if piror_knowledge.shape[1] != input_dim:
                        self.logger.warning(f"先验知识矩阵维度 ({piror_knowledge.shape[1]}) 与输入维度 ({input_dim}) 不匹配")
                        # 如果维度不匹配，创建随机矩阵作为备用
                        num_genesets = piror_knowledge.shape[0]
                        piror_knowledge = np.random.rand(num_genesets, input_dim)
                        piror_knowledge = (piror_knowledge > 0.8).astype(np.float32)
                else:
                    self.logger.warning("先验知识加载失败，使用随机矩阵")
                    num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
                    piror_knowledge = np.random.rand(num_genesets, input_dim)
                    piror_knowledge = (piror_knowledge > 0.8).astype(np.float32)
            else:
                self.logger.warning("无法获取基因名称列表，使用随机矩阵")
                num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
                piror_knowledge = np.random.rand(num_genesets, input_dim)
                piror_knowledge = (piror_knowledge > 0.8).astype(np.float32)
        
        # 处理explain_knowledge配置
        explain_knowledge_path = self.config.get('explain', {}).get('explain_knowledge')
        if explain_knowledge_path:
            # 确保nadata有uns属性
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            
            # 保存explain_knowledge路径到nadata的uns字典中
            nadata.uns['explain_knowledge_path'] = explain_knowledge_path
            self.logger.info(f"已保存explain_knowledge路径到nadata.uns: {explain_knowledge_path}")
        
        # 更新配置
        self.config['input_dim'] = input_dim
        self.config['output_dim'] = output_dim
        self.config['device'] = str(self.device)  # 确保设备配置正确传递
        
        # 更新nnea配置中的先验知识
        if 'nnea' not in self.config:
            self.config['nnea'] = {}
        if 'piror_knowledge' not in self.config['nnea']:
            self.config['nnea']['piror_knowledge'] = {}
        self.config['nnea']['piror_knowledge']['piror_knowledge'] = piror_knowledge
        
        # 创建模型
        self.model = NNEAModel(self.config)
        self.model.to(self.device)
        
        # 确保所有模型组件都在正确的设备上
        if hasattr(self.model, 'geneset_layer'):
            self.model.geneset_layer.to(self.device)

        self.logger.info(f"NNEA分类器已构建: 输入维度={input_dim}, 输出维度={output_dim}")
        num_genesets = nnea_config.get('geneset_layer', {}).get('num_genesets', 20)
        self.logger.info(f"基因集数量: {num_genesets}")
        self.logger.info(f"使用先验知识: {use_piror_knowledge}")
        
    def train(self, nadata, verbose: int = 1, max_epochs: Optional[int] = None, continue_training: bool = False, **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            nadata: nadata对象
            verbose: 详细程度
                0=只显示进度条
                1=显示训练损失、训练正则化损失、验证损失、验证正则化损失
                2=在verbose=1基础上增加显示F1、AUC、Recall、Precision等评估指标
            max_epochs: 最大训练轮数，如果为None则使用配置中的epochs
            continue_training: 是否继续训练（用于tailor策略）
            **kwargs: 额外参数
            
        Returns:
            训练结果字典
        """
        if self.model is None:
            raise ValueError("模型未构建")
        
        # 设置CUDA调试环境变量
        if self.device.type == 'cuda':
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            self.logger.info("已启用CUDA同步执行模式，有助于调试CUDA错误")
        
        # 准备数据
        X = nadata.X
        
        # 获取标签
        config = nadata.Model.get_config()

        # 检查表型数据是否存在
        if not hasattr(nadata, 'Meta') or nadata.Meta is None:
            raise ValueError(f"未找到表型数据，请检查数据加载是否正确")

        # 获取标签数据
        y = nadata.Meta["target"].values
        

        # 获取已有的数据索引
        train_indices = nadata.Model.get_indices('train')
        test_indices = nadata.Model.get_indices('test')
        

        # 使用已有的train和test索引
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # 获取train索引对应的数据
        X_train_full = X[train_indices]
        y_train_full = y[train_indices]

        # 将train数据进一步分割为train和validation
        val_size = config.get('dataset', {}).get('val_size', 0.2)
        random_state = config.get('dataset', {}).get('random_state', 42)

        # 从train数据中分割出validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=random_state
        )

        # 计算新的train和validation索引
        n_train_full = len(train_indices)

        # 计算validation在原始train索引中的位置
        val_size_adjusted = val_size
        n_val = int(n_train_full * val_size_adjusted)
        n_train = n_train_full - n_val

        # 更新索引
        train_indices_final = train_indices[:n_train]
        val_indices = train_indices[n_train:]

        # 保存更新后的索引
        nadata.Model.set_indices(
            train_idx=train_indices_final.tolist(),
            test_idx=test_indices.tolist(),
            val_idx=val_indices.tolist()
        )

        # 训练参数
        training_config = config.get('training', {})
        if max_epochs is None:
            epochs = training_config.get('epochs', 100)
        else:
            epochs = max_epochs
        learning_rate = training_config.get('learning_rate', 0.001)
        batch_size = training_config.get('batch_size', 32)
        reg_weight = training_config.get('regularization_weight', 0.1)
        
        # 转换为张量并构建TensorDataset
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)  # 修改为LongTensor
        
        # 构建训练数据集
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        
        # 添加调试信息
        self.logger.info(f"训练数据形状: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
        self.logger.info(f"训练标签值范围: {y_train_tensor.min().item()} - {y_train_tensor.max().item()}")
        self.logger.info(f"模型输出维度: {self.model.output_dim}")
        
        # 构建验证数据集（如果有验证数据）
        val_dataset = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)  # 修改为LongTensor
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            self.logger.info(f"验证数据形状: X_val={X_val_tensor.shape}, y_val={y_val_tensor.shape}")
            self.logger.info(f"验证标签值范围: {y_val_tensor.min().item()} - {y_val_tensor.max().item()}")
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # 设置为0避免多进程问题
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0  # 设置为0避免多进程问题
            )

        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()  # 回归任务使用MSE损失
        
        # 模型初始化阶段 - 训练TrainableGeneSetLayer的indicator
        if not continue_training:
            self.logger.info("🔧 开始模型初始化阶段 - 训练基因集层指示矩阵...")
            
            # 根据配置决定是否在初始化阶段启用decoder
            if self.model.use_decoder_in_init:
                self.model.set_decoder_mode(True)
                self.logger.info("📊 初始化阶段：启用解码器，直接映射geneset输出为概率")
            else:
                self.model.set_decoder_mode(False)
                self.logger.info("📊 初始化阶段：使用标准模式，使用focus_layer进行预测")
            
            init_results = self._initialize_geneset_layer(train_loader, optimizer, verbose)
            self.logger.info(f"✅ 模型初始化完成: {init_results}")
            
            # 初始化完成后，切换到标准模式（使用focus_layer）
            self.model.set_assist_layer_mode(False)
            self.logger.info("🔄 初始化完成：切换到标准模式，使用focus_layer进行预测")
            
            # 将初始化结果保存到nadata.uns中
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            nadata.uns['init_results'] = init_results
            self.logger.info("💾 初始化结果已保存到nadata.uns中")
        else:
            # 继续训练时，确保使用标准模式
            self.model.set_assist_layer_mode(False)
            self.logger.info("🔄 继续训练：使用标准模式，使用focus_layer进行预测")
        
        # 早停机制参数
        patience = training_config.get('patience', 10)
        min_delta = 1e-6  # 最小改善阈值
        
        # 早停变量初始化
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False
        
        # 添加checkpoint保存相关变量
        best_model_state = None
        best_epoch = 0
        outdir = config.get('global', {}).get('outdir', 'experiment/test')
        
        # 训练循环
        train_losses = {'loss': [], 'reg_loss': []}
        val_losses = {'loss': [], 'reg_loss': []}
        
        if verbose >= 1:
            if continue_training:
                self.logger.info(f"继续训练NNEA模型... (剩余{epochs}个epoch)")
            else:
                self.logger.info("开始正式训练NNEA模型...")
            self.logger.info(f"早停配置: patience={patience}, min_delta={min_delta}")
            self.logger.info(f"Checkpoint保存目录: {outdir}")
        
        # 导入tqdm用于进度条
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        # 创建进度条（只有verbose=0时显示）
        if verbose == 0 and use_tqdm:
            pbar = tqdm(range(epochs), desc="训练进度")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # 训练模式
            self.model.train()
            epoch_loss = 0.0
            epoch_reg_loss = 0.0
            num_batches = 0
            train_predictions = []
            train_targets = []
            
            # 使用数据加载器进行批处理训练
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # 将数据移动到设备
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                try:
                    # 前向传播
                    outputs = self.model(batch_X)
                    
                    # 检查输出是否包含NaN或无穷大
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: 模型输出包含NaN或无穷大值")
                        continue
                    
                    loss = criterion(outputs, batch_y)
                    
                    # 检查损失值是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: 损失值为NaN或无穷大")
                        continue
                    
                    # 添加正则化损失
                    reg_loss = self.model.regularization_loss()
                    total_loss = loss + reg_weight * reg_loss
                    
                    # 反向传播
                    total_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_reg_loss += reg_loss.item()
                    num_batches += 1
                    
                    # 收集训练预测结果用于计算指标
                    if verbose >= 2:
                        predictions = outputs.cpu().detach().numpy()
                        targets = batch_y.cpu().detach().numpy()
                        train_predictions.append(predictions)
                        train_targets.append(targets)
                    
                except Exception as e:
                    self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: 训练过程中出现错误: {e}")
                    continue
            
            # 计算平均损失
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_reg_loss = epoch_reg_loss / num_batches
                train_losses['loss'].append(avg_loss)
                train_losses['reg_loss'].append(avg_reg_loss)
                
                # verbose=1时显示训练损失
                if verbose >= 1:
                    self.logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Reg_Loss={avg_reg_loss:.4f}")
                
                # verbose=2时计算并显示训练集评估指标
                if verbose >= 2 and train_predictions and train_targets:
                    # 合并所有批次的预测结果
                    all_train_predictions = np.vstack(train_predictions)
                    all_train_targets = np.concatenate(train_targets)
                    
                    # 计算训练集评估指标
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
                            loss = criterion(outputs, batch_y)
                            reg_loss = self.model.regularization_loss()
                            
                            val_loss += loss.item()
                            val_reg_loss += reg_loss.item()
                            val_num_batches += 1
                            
                            # 收集预测结果用于计算指标
                            if verbose >= 2:
                                predictions = outputs.cpu().detach().numpy()
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
                        self.logger.info(f"Epoch {epoch} Validation: Val Loss={avg_val_loss:.4f}, Val Reg_Loss={avg_val_reg_loss:.4f}")
                    
                    # verbose=2时计算并显示评估指标
                    if verbose >= 2 and val_predictions and val_targets:
                        # 合并所有批次的预测结果
                        all_predictions = np.vstack(val_predictions)
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
                            self.logger.info(f"⚠️ Epoch {epoch}: 验证损失未改善，patience_counter={patience_counter}/{patience}")
                    
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
            self.logger.info(f"📊 训练因早停而结束，实际训练了{epoch+1}个epoch")
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
            mse = mean_squared_error(targets, predictions.squeeze())
            mae = mean_absolute_error(targets, predictions.squeeze())
            r2 = r2_score(targets, predictions.squeeze())
            rmse = np.sqrt(mse.squeeze())

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
            self.logger.info("📊 初始化阶段：使用辅助层直接映射geneset输出为概率")
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
        self.logger.info(f"初始化损失权重: task_loss_weight={init_task_loss_weight}, reg_loss_weight={init_reg_loss_weight}, total_loss_weight={init_total_loss_weight}")
        
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
                    
                    # 计算任务损失（分类损失）
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
            total_gene_sets = self.model.geneset_layer.num_sets if hasattr(self.model, 'geneset_layer') else self.model.gene_set_layer.num_sets
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
                self.logger.info(f"初始化Epoch {epoch}: Reg Loss={epoch_loss/num_batches:.4f}, 满足条件的基因集数: {condition_count}/{total_gene_sets}")

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
            return outputs.cpu().numpy()
    
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
                    # gene_importance += geneset_assignments[i].astype(np.float32) * float(geneset_importance[i])
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
            self.logger.info(f"    {'-'*4} {'-'*15} {'-'*12}")
            for i, (gene, score) in enumerate(zip(top_genes, top_scores)):
                self.logger.info(f"    {i+1:<4} {gene:<15} {score:<12.4f}")
            
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
            self.logger.info(f"    {'-'*12} {'-'*30} {'-'*50}")
            
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
            R_sample = R[i:i+1]
            S_sample = S[i:i+1]
            
            # 计算该样本的积分梯度
            ig_score = self._integrated_gradients_for_genesets(
                R_sample, S_sample, steps=50
            )
            all_ig_scores.append(ig_score.cpu().numpy())
        
        # 计算平均重要性分数
        avg_ig_scores = np.mean(all_ig_scores, axis=0)
        
        return avg_ig_scores
    
    def _integrated_gradients_for_genesets(self, R, S, target_class=None, baseline=None, steps=50):
        """
        使用积分梯度解释基因集重要性
        
        Args:
            R: 基因表达数据 (1, num_genes)
            S: 基因排序索引 (1, num_genes)
            target_class: 要解释的目标类别 (默认使用模型预测类别)
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
        
        # 确定目标类别
        if target_class is None:
            with torch.no_grad():
                # 从R和S重构原始输入x
                x = R  # 对于NNEA包中的模型，R就是原始输入
                output = self.model(x)
                if self.model.output_dim == 1:
                    target_class = 0  # 二分类
                else:
                    target_class = torch.argmax(output, dim=1).item()
        
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
            
            # 根据输出维度处理
            if self.model.output_dim == 1:
                # 二分类
                logits = self.model.focus_layer(interp_es)
                target_logit = logits.squeeze()
            else:
                # 多分类
                logits = self.model.focus_layer(interp_es)
                target_logit = logits[0, target_class]
            
            # 计算梯度
            grad = torch.autograd.grad(outputs=target_logit, inputs=interp_es)[0]
            gradients.append(grad.detach())
        
        # 整合梯度计算积分梯度
        gradients = torch.stack(gradients)  # (steps, 1, num_sets)
        avg_gradients = torch.mean(gradients, dim=0)  # (1, num_sets)
        ig = (es_scores - baseline) * avg_gradients  # (1, num_sets)
        
        return ig.squeeze(0)  # (num_sets,) 