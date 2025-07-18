import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from torch import nn
import torch.nn.functional as F
import joblib

import shutil
from utils.train_utils import cox_loss, BuildOptimizer, LoadModel
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score
)

class CrossTrainer(object):

    def __init__(self, trainer_config, model_config, global_config, loader):
        self.model_config = {**global_config, **model_config, **trainer_config}
        self.loader = loader
        self.best_params = None  # 存储最佳参数
        self.best_score = -float('inf')  # 存储最佳得分

    def train_single_model(self, fold_data):

        model = LoadModel(self.model_config, self.loader)

        # 创建Trainer并训练
        trainer = Trainer(self.model_config, model, fold_data)
        trainer.train()

        # 返回验证集性能
        return trainer.get_validation_metric()

    def save_cv_results(self, results):
        """保存交叉验证结果"""
        df = pd.DataFrame([{
            'params': str(r[0]),
            'avg_score': r[1],
            'fold_scores': str(r[2])
        } for r in results])
        df.to_csv(os.path.join(self.model_config['checkpoint_dir'], 'cv_results.csv'), index=False)

    def train(self):

        param_grid = {
            "lr" : self.model_config['lr'],
            "weight_decay" : self.model_config['weight_decay'],
            "classifier_dropout": self.model_config['classifier_dropout'],
            "num_sets": range(self.model_config['num_sets'][0],
                              self.model_config['num_sets'][1],
                              self.model_config['num_sets'][2]),
            "batch_size" : self.model_config['batch_size'],
        }

        param_samples = list(ParameterSampler(param_grid, n_iter=self.model_config['n_iter']))
        results = []

        for params in param_samples:
            print(f"\nEvaluating hyperparameters: {params}")
            fold_scores = []
            self.model_config.update(params)
            for item in self.loader.cv_loaders:

                score = self.train_single_model(item)
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            results.append((params, avg_score, fold_scores))

            # 更新最佳参数
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_params = params
                best_fold = np.argmax(fold_scores)
                print(f"🔥 New best params! Score: {avg_score:.4f}")


        print(f"\n🚀 Training final model with best params: {self.best_params}")
        self.model_config.update(self.best_params)
        final_model = LoadModel(self.model_config, self.loader)
        final_trainer = Trainer(self.model_config, final_model, self.loader.cv_loaders[best_fold])
        final_trainer.train()
        test_loader = torch.utils.data.DataLoader(
            self.loader.test_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False
        )
        final_trainer.evaluate(loader=test_loader)

        print(classification_report(final_trainer.all_targets, final_trainer.all_predictions))

        with open(os.path.join(self.model_config['checkpoint_dir'], "test_result.txt"), 'w') as f:
            f.write(classification_report(final_trainer.all_targets, final_trainer.all_predictions))

        # 保存交叉验证结果
        results.append((self.best_params, "<-best param; best fold->", best_fold))
        self.save_cv_results(results)
class Trainer(object):

    def __init__(self, config, model, loader):

        self.config = config
        self.model = model
        self.loader = loader

        if config['model'] == "nnea":
            self.init_nnea()

    def init_nnea(self):
        # self.loader = self.loader.torch_loader

        self.train_loader = torch.utils.data.DataLoader(
            self.loader['train'],
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.loader['valid'],
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 裁剪梯度
        self.scheduler, self.optimizer = BuildOptimizer(params=self.model.parameters(), config=self.config)


        if self.config['task'] == "umap":
            self.umap_loss = UMAP_Loss(
                n_neighbors=self.config['n_neighbors'],
                min_dist=self.config['min_dist']
            )


    def evaluate(self, loader):

        self.model.eval()
        total_samples = 0
        pos_samples = 0
        neg_samples = 0
        pos_risk = 0
        neg_risk = 0
        correct = 0
        mse_loss = 0.0
        mae_loss = 0.0
        recon_loss = 0.0
        silhouette_loss = 0.0
        all_predictions = []  # 存储所有预测值
        all_targets = []  # 存储所有真实标签

        with torch.no_grad():
            for batch_data in loader:

                device = self.config['device']
                batch_R, batch_S = batch_data[0].to(device),  batch_data[1].to(
                    device)

                # 前向传播
                logits = self.model(batch_R, batch_S)

                total_samples += batch_data[0].size(0)

                if self.config['task'] in ['classification']:

                    _, predicted = torch.max(logits, 1)
                    batch_y = batch_data[2]
                    correct += (predicted == batch_y.squeeze().to(self.config["device"])).sum().item()

                    # 累积预测值和标签
                    all_predictions.append(predicted.cpu())
                    all_targets.append(batch_y.cpu())

                elif self.config['task'] in ['cox']:
                    events = batch_data[3]
                    pos_samples += sum(events == 1)
                    neg_samples += sum(events == 0)
                    pos_risk += torch.sum(logits[events == 1])
                    neg_risk += torch.sum(logits[events == 0])

                elif self.config['task'] in ['regression']:
                    predictions = logits.squeeze()
                    batch_y = batch_data[2].to(device, dtype=torch.float)

                    # 计算 MSE 和 MAE
                    mse_loss += nn.MSELoss()(predictions, batch_y).item() * len(batch_y)
                    mae_loss += nn.L1Loss()(predictions, batch_y).item() * len(batch_y)

                    # 累积预测值和标签
                    all_predictions.append(predictions.cpu())
                    all_targets.append(batch_y.cpu())

                elif self.config['task'] in ['autoencoder']:
                    enc_exp = logits
                    norm_exp = batch_data[2].to(device, dtype=torch.float)
                    batch_recon_loss = F.mse_loss(enc_exp, norm_exp).item()
                    recon_loss += batch_recon_loss * batch_R.size(0)

                elif self.config['task'] in ['umap']:
                    embeddings = logits.cpu().detach()
                    labels = batch_data[3].cpu().detach()# 假设batch_data[1]是细胞类型
                    all_predictions.append(embeddings)
                    all_targets.append(labels)


            if self.config['task'] in ['classification']:
                self.accuracy = correct / total_samples

                # 计算混淆矩阵和F1分数
                all_preds = torch.cat(all_predictions).numpy()
                all_targets = torch.cat(all_targets).numpy()

                self.all_predictions = all_preds
                self.all_targets = all_targets

                # 计算每个类别的精确率、召回率和F1分数
                self.classification_report = classification_report(
                    all_targets, all_preds,
                    target_names=self.config['class_names'],
                    output_dict=True,
                    zero_division=0  # 或 1，根据需求设置
                )
                self.macro_f1 = self.classification_report['macro avg']['f1-score']
                self.weighted_f1 = self.classification_report['weighted avg']['f1-score']


            if self.config['task'] in ['autoencoder']:
                self.recon_loss = recon_loss / total_samples

            elif self.config['task'] in ['umap']:

                all_preds = torch.cat(all_predictions).numpy()
                all_targets = torch.cat(all_targets).numpy()

                self.silhouette_loss = silhouette_score(all_preds, all_targets)

            elif self.config['task'] in ['cox']:
                self.high_risk = pos_risk / total_samples
                self.low_risk = neg_risk / total_samples
                self.accuracy = self.high_risk - self.low_risk

            elif self.config['task'] == 'regression':
                self.mse = mse_loss / total_samples
                self.mae = mae_loss / total_samples

                all_preds = torch.cat(all_predictions).numpy()
                all_targets = torch.cat(all_targets).numpy()

                pearson_corr = np.corrcoef(all_preds, all_targets)[0, 1]
                self.pearson = pearson_corr if not np.isnan(pearson_corr) else 0.0

    def get_validation_metric(self):
        """获取验证集性能指标"""

        # 根据任务类型返回关键指标
        if self.config['task'] == "classification":
            return self.macro_f1
        elif self.config['task'] == "cox":
            return self.accuracy
        elif self.config['task'] == "regression":
            return -self.mse  # 负MSE（越大越好）
        elif self.config['task'] == "umap":
            return self.silhouette_loss

    def get_task_loss(self, logits, batch_data, device):

        if self.config['task'] in ['classification']:
            log_probs = logits
            batch_y = batch_data[2].to(device, dtype=torch.long).squeeze()
            return nn.NLLLoss()(log_probs, batch_y)

        elif self.config['task'] in ['cox']:
            risks = logits
            times = batch_data[2].to(device)
            events = batch_data[3].to(device)
            return cox_loss(risks, times, events)

        elif self.config['task'] == 'regression':
            predictions = logits.squeeze()  # 确保输出是1D向量
            batch_y = batch_data[2].to(device, dtype=torch.float)
            return nn.MSELoss()(predictions, batch_y)

        elif self.config['task'] == 'autoencoder':
            enc_exp = logits  # 确保输出是1D向量
            norm_exp = batch_data[2].to(device, dtype=torch.float)

            return F.mse_loss(enc_exp, norm_exp)

        elif self.config['task'] == 'umap':
            embeddings = logits
            inputs = batch_data[2].to(device, dtype=torch.float)
            return self.umap_loss(embeddings, inputs)

    def calculate_loss(self, loader):

        self.model.eval()
        total_loss = 0.0

        # for batch_R, batch_S, batch_y in self.loader:
        for batch_data in loader:
            device = self.config['device']
            batch_R, batch_S = batch_data[0].to(device), batch_data[1].to(
                device)

            # 前向传播
            logits = self.model(batch_R, batch_S)

            # 计算分类损失
            task_loss = self.get_task_loss(logits, batch_data, device=self.config['device'])

            # 计算正则化损失
            reg_loss = self.model.regularization_loss()

            # 总损失（分类损失+正则化损失）
            total_batch_loss = task_loss + reg_loss

            total_loss += total_batch_loss.item()

        avg_loss = total_loss / len(self.loader)

        return avg_loss, task_loss, reg_loss

    def one_epoch(self):
        self.model.train()
        total_loss = 0.0

        # for batch_R, batch_S, batch_y in self.loader:
        for batch_data in self.train_loader:

            device = self.config['device']
            batch_R, batch_S = batch_data[0].to(device),  batch_data[1].to(
                device)

            self.optimizer.zero_grad()

            # 前向传播
            logits = self.model(batch_R, batch_S)

            # 计算分类损失
            task_loss = self.get_task_loss(logits, batch_data, device=self.config['device'])

            # 计算正则化损失
            reg_loss = self.model.regularization_loss()

            # 总损失（分类损失+正则化损失）
            total_batch_loss = task_loss + reg_loss

            # 反向传播
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()

            total_loss += total_batch_loss.item()

        avg_loss = total_loss / len(self.loader)

        return avg_loss, task_loss, reg_loss

    def save_checkpoint(self):

        checkpoint_dir = self.config['checkpoint_dir']
        if not os.path.exists(checkpoint_dir):  # 先检查是否存在
                os.makedirs(checkpoint_dir)
                source_file = self.config['toml_path']
                dest_path = os.path.join(checkpoint_dir, os.path.basename(source_file))
                shutil.copy2(source_file, dest_path)  # 复制文件并保留元数据[6,8](@ref)

        if self.config['model'] in ['nnea']:
            torch.save(self.model.state_dict(), os.path.join(self.config['checkpoint_dir'], "_checkpoint.pt"))

        elif self.config['model'] in ['LR']:
            test_metrics, test_pred = self.evaluate_model(self.model, self.loader.X_test, self.loader.y_test)
            print("best params: %s" % self.model.get_params)
            print("best metrics: %s" % test_metrics)
            print(classification_report(self.loader.y_test, test_pred))

            joblib.dump(self.model,  os.path.join(self.config['checkpoint_dir'], "_checkpoint.pkl"))
            with open(os.path.join(self.config['checkpoint_dir'], "test_result.txt"), 'w') as f:
                f.write(f"Best Params: {self.model.get_params}\n")
                f.write(f"Test Metrics: {test_metrics}\n")
                f.write("Classification Report:\n")
                f.write(classification_report(self.loader.y_test, test_pred))

            results_df = pd.DataFrame({
                'true_label': self.loader.y_test,
                'predicted_label': test_pred
            })
            results_df.to_csv(os.path.join(self.config['checkpoint_dir'], "'predictions.csv"), index=False)

    def save_model(self, epoch):

        # 任务类型与监控指标的映射
        if self.config['task'] == "cox":
            current_metric = self.accuracy
        elif self.config['task'] == "classification":
            current_metric = self.macro_f1
        elif self.config['task'] == "regression":
            current_metric = -self.mse  # 负值方便统一逻辑（越大越好）
        elif self.config['task'] == "autoencoder":
            current_metric = -self.recon_loss
        elif self.config['task'] == "umap":
            current_metric = -self.silhouette_loss

        improved = current_metric > self.best_metric

        if improved:
            self.best_metric = current_metric
            self.save_checkpoint()

        return improved
    def print_process(self, epoch, task_loss, reg_loss, avg_loss):

        if self.config['task'] in ['classification']:

            info = (f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {avg_loss:.4f}, "
                  f"Task Loss: {task_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, "
                  f"Train accuracy: {self.accuracy:.4f} "
                  f"Train macro f1: {self.macro_f1:.4f}, "
                  f"Train weighted f1: {self.weighted_f1:.4f}, "
                  )
            self.evaluate(loader=self.valid_loader)

            info += (f"Val accuracy: {self.accuracy:.4f} "
                  f"Val macro f1: {self.macro_f1:.4f}, "
                  f"Val weighted f1: {self.weighted_f1:.4f}")

            print(info)

        elif self.config['task'] in ['cox']:

            info = (f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {avg_loss:.4f}, "
                  f"Task Loss: {task_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, "
                  f"Train diff risk: {self.accuracy:.4f}, "
                  f"Train high risk: {self.high_risk:.4f}, Train low risk: {self.low_risk:.4f}, ")

            self.evaluate(loader=self.valid_loader)

            info += (
                  f"Val diff risk: {self.accuracy:.4f}, "
                  f"Val high risk: {self.high_risk:.4f}, Val low risk: {self.low_risk:.4f}")

            print(info)


        elif self.config['task'] in ['regression']:
            info = (f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {avg_loss:.4f}, "
                  f"Task Loss: {task_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, "
                  f"Train mse: {self.mse:.4f}, Train mae: {self.mae:.4f}, Train pcc: {self.pearson:.4f}, ")

            self.evaluate(loader=self.valid_loader)
            info += (f"Val mse: {self.mse:.4f}, Val mae: {self.mae:.4f}, Val pcc: {self.pearson:.4f}")

            print(info)

        elif self.config['task'] in ['autoencoder']:
            info = (f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {avg_loss:.4f}, "
                  f"Task Loss: {task_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, "
                  f"Train rec Loss: {self.recon_loss:.4f}, ")

            self.evaluate(loader=self.valid_loader)
            info += (f"Train rec Loss: {self.recon_loss:.4f}")
            print(info)

        elif self.config['task'] in ['umap']:
            info = (f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {avg_loss:.4f}, "
                  f"Task Loss: {task_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, "
                  f"Silhouette Loss: {self.silhouette_loss:.4f}, ")

            self.evaluate(loader=self.valid_loader)
            info += (f"Silhouette Loss: {self.silhouette_loss:.4f}")
            print(info)

    def train_nnea(self):

        # 初始化早停相关变量（同时监控训练损失和验证指标）
        self.best_metric = -float('inf') if self.config['task'] in ['classification', 'cox'] else float('inf')
        self.best_avg_loss = float('inf')  # 新增：追踪最佳训练损失
        self.patience_counter_metric = 0  # 验证指标无改善的计数器
        self.patience_counter_loss = 0  # 训练损失无改善的计数器

        for epoch in range(self.config['num_epochs']):

            avg_loss, task_loss, reg_loss = self.one_epoch()
            self.evaluate(loader=self.train_loader)
            self.scheduler.step(avg_loss)

            # 打印训练信息
            self.print_process(epoch, task_loss, reg_loss, avg_loss)

            # 关键修改：同时监控训练损失和验证指标
            improved_metric = self.save_model(epoch)  # 验证指标是否提升

            avg_loss, task_loss, reg_loss = self.calculate_loss(loader = self.valid_loader)
            improved_loss = avg_loss < self.best_avg_loss  # 训练损失是否降低

            # 更新最佳损失
            if improved_loss:
                self.best_avg_loss = avg_loss
                self.patience_counter_loss = 0
            else:
                self.patience_counter_loss += 1

            # 更新早停计数器（验证指标）
            if improved_metric:
                self.patience_counter_metric += 1
            else:
                self.patience_counter_metric = 0


            # 早停条件：训练损失或验证指标连续恶化
            # if (self.patience_counter_metric >= self.config['patience_metric'] and
            #         self.patience_counter_loss >= self.config['patience_loss']):
            if (self.patience_counter_loss >= self.config['patience_loss']):
                print(f"Early stopping at epoch {epoch}")
                break

    def evaluate_model(self, model, X, y):
        """评估模型性能并返回指标字典"""
        pred = model.predict(X)

        if self.config['task'] == 'classification':
            metrics = {
                "Accuracy": accuracy_score(y, pred),
                "F1": f1_score(y, pred, average='weighted'),
            }
            try:
                proba = model.predict_proba(X)[:, 1]
                metrics["AUC"] = roc_auc_score(y, proba)
            except:
                metrics["AUC"] = None
        elif self.config['task'] == 'regression':  # 回归任务
            metrics = {
                "MSE": mean_squared_error(y, pred),
                "MAE": mean_absolute_error(y, pred),
                "R2": r2_score(y, pred)
            }
        return metrics, pred


    def train(self):

        if self.config['model'] in ["nnea"]:
            self.train_nnea()

        elif self.config['model'] in ["LR"]:

            if self.config['train_mod'] == "cross_validation":
                searcher = RandomizedSearchCV(
                    estimator=self.model["model"],
                    param_distributions=self.model["params"],
                    n_iter=15,
                    cv=self.loader.cv,
                    scoring=self.config["scoring"],
                    n_jobs=-self.config['n_jobs'],
                    verbose=self.config['verbose'],
                    random_state=self.config['seed']
                )

                searcher.fit(self.loader.X_train, self.loader.y_train)


                # 保存最佳模型
                self.model = searcher.best_estimator_
                self.save_checkpoint()


            elif self.config['train_mod'] == "one_split":
                self.model.fit(self.loader.X_train, self.loader.y_train)
                test_metrics, test_pred = self.evaluate_model(self.model, self.loader.X_test, self.loader.y_test)
                print(classification_report(self.loader.y_test, test_pred))




class UMAP_Loss(nn.Module):
    def __init__(self, n_neighbors=15, min_dist=0.1):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.a, self.b = self._find_ab_params(min_dist)

    def _find_ab_params(self, min_dist):
        a = 1.0 / (min_dist ** 2)  # 柯西分布参数计算[6](@ref)
        b = torch.log(torch.tensor(2.0))
        return a, b

    def forward(self, embeddings, data):
        # 1. 动态调整邻居数（关键修复！）
        n_samples = data.size(0)
        effective_k = min(self.n_neighbors + 1, n_samples)  # 确保不超过样本数

        # 2. 高维空间相似度
        distances = torch.cdist(data, data)

        # 处理小batch情况
        if effective_k <= 1:  # 无足够邻居可计算
            return torch.tensor(0.0, device=data.device, requires_grad=True)

        # 获取最近的k个点（包括自身）
        _, indices = torch.topk(distances, effective_k, largest=False)

        # 3. 构建邻居掩码（排除自身）
        neighbor_mask = torch.zeros_like(distances, device=data.device)
        row_idx = torch.arange(n_samples).view(-1, 1).expand(-1, effective_k - 1)
        neighbor_mask[row_idx, indices[:, 1:]] = 1  # 跳过自身（索引0）

        # 4. 对称化邻接矩阵（UMAP核心）
        neighbor_mask = neighbor_mask.float()
        sym_mask = (neighbor_mask + neighbor_mask.t()) > 0  # 取并集[1,4](@ref)

        # 5. 计算高维相似度（使用局部自适应尺度）
        local_scale = distances.gather(1, indices[:, 1:2]).mean(dim=1)  # 自适应σ[4](@ref)
        p_ij = sym_mask * torch.exp(-distances ** 2 / (local_scale.unsqueeze(1) * local_scale))
        p_ij = p_ij / torch.clamp(p_ij.sum(dim=1), min=1e-12).unsqueeze(1)

        # 6. 低维空间相似度
        low_dim_dist = torch.cdist(embeddings, embeddings)
        q_ij = 1.0 / (1 + self.a * low_dim_dist ** (2 * self.b))  # UMAP曲线拟合[1,6](@ref)

        # 7. 交叉熵损失（原始UMAP设计）[1,4,6](@ref)
        loss_pos = p_ij * torch.log(q_ij + 1e-7)
        loss_neg = (1 - p_ij) * torch.log(1 - q_ij + 1e-7)
        return -(loss_pos + loss_neg).mean()