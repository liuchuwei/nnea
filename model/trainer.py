import os

import torch
from torch import nn

import shutil

class Trainer(object):

    def __init__(self, config, model, loader):

        self.config = config
        self.model = model
        self.loader = loader.torch_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪梯度
        self.classification_loss = nn.NLLLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5)

    def evaluate(self):

        self.model.eval()

        with torch.no_grad():
            correct = 0
            total_samples = 0

            for batch_R, batch_S, batch_y in self.loader:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch_R, batch_S, batch_y = batch_R.to(device), batch_S.to(
                    device), batch_y.to(device)
                log_probs = self.model(batch_R, batch_S)

                _, predicted = torch.max(log_probs, 1)
                correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)

            accuracy = correct / total_samples

        return accuracy

    def one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch_R, batch_S, batch_y in self.loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch_R, batch_S, batch_y = batch_R.to(device),  batch_S.to(
                device), batch_y.to(device)

            self.optimizer.zero_grad()

            # 前向传播
            log_probs = self.model(batch_R, batch_S)

            # 计算分类损失
            class_loss = self.classification_loss(log_probs, batch_y)

            # 计算正则化损失
            reg_loss = self.model.regularization_loss()

            # 总损失（分类损失+正则化损失）
            total_batch_loss = class_loss + reg_loss

            # 反向传播
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()

            total_loss += total_batch_loss.item()

        avg_loss = total_loss / len(self.loader)

        return avg_loss, class_loss, reg_loss

    def save_checkpoint(self, epoch):

        checkpoint_dir = self.config['checkpoint_dir']
        if not os.path.exists(checkpoint_dir):  # 先检查是否存在
                os.makedirs(checkpoint_dir)
                source_file = self.config['path']
                dest_path = os.path.join(checkpoint_dir, os.path.basename(source_file))
                shutil.copy2(source_file, dest_path)  # 复制文件并保留元数据[6,8](@ref)

        torch.save(self.model.state_dict(), self.config['check_point'])

    def train(self):

        best_accuracy = 0

        for epoch in range(self.config['num_epochs']):

            avg_loss, class_loss, reg_loss = self.one_epoch()
            accuracy = self.evaluate()
            self.scheduler.step(avg_loss)

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                self.save_checkpoint(epoch)
                # torch.save(self.model.state_dict(), self.config['check_point'])

            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {avg_loss:.4f}, "
                  f"Class Loss: {class_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, "
                  f"Accuracy: {accuracy:.4f}")
