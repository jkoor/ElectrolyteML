import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
from typing import Optional, Union
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error

# 假设模型定义在 .models 中
from .models import ElectrolyteMLP, ElectrolyteTransformer


class TrainingWorkflow:
    """
    一个管理机器学习模型训练、评估和预测全过程的工作流类。
    """

    def __init__(
        self,
        model: Union[ElectrolyteMLP, ElectrolyteTransformer],
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        log_dir: str = "runs",
        optimizer_name: str = "adamw",
    ):
        """
        初始化工作流。

        Args:
            model: PyTorch模型实例 (ElectrolyteMLP 或 ElectrolyteTransformer)。
            train_dataset: 训练数据集。
            val_dataset: 验证数据集。
            test_dataset: 测试数据集 (可选)。
            batch_size (int): 批处理大小。
            lr (float): 学习率。
            weight_decay (float): 权重衰减 (L2正则化)。
            device (str, optional): 指定设备 ('cuda', 'cpu')。如果为None，则自动检测。
            log_dir (str): 保存日志和模型检查点的目录。
            optimizer_name (str): 优化器名称 ("adamw", "adam" 或 "sgd")。
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.log_dir = log_dir

        # 自动检测设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)

        # 初始化DataLoader
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        if self.test_dataset:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        # 初始化优化器、损失函数和学习率调度器
        if optimizer_name == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.criterion = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=10
        )

        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

    def _run_epoch(self, loader: DataLoader, is_train: bool = True):
        """运行一个epoch的内部辅助函数。"""
        self.model.train() if is_train else self.model.eval()
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()

        for batch_data in loader:
            if len(batch_data) == 3:  # Transformer with mask
                features, targets, padding_mask = batch_data
                features = features.to(self.device)
                targets = targets.to(self.device) 
                padding_mask = padding_mask.to(self.device)
                
                outputs = self.model(features, src_padding_mask=padding_mask)
            else:  # MLP without mask
                features, targets = batch_data
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)

            with torch.set_grad_enabled(is_train):
                loss = self.criterion(outputs.squeeze(), targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        epoch_duration = time.time() - start_time
        return epoch_loss, epoch_duration

    def train(self, num_epochs: int, early_stopping_patience: int = 20):
        """
        执行完整的训练循环。

        Args:
            num_epochs (int): 训练的总轮次。
            early_stopping_patience (int): 早停的耐心轮次。
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0

        print("Starting training...")
        for epoch in range(num_epochs):
            train_loss, train_duration = self._run_epoch(
                self.train_loader, is_train=True
            )
            val_loss, val_duration = self._run_epoch(self.val_loader, is_train=False)

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} ({train_duration:.2f}s) | "
                f"Val Loss: {val_loss:.4f} ({val_duration:.2f}s)"
            )

            self.scheduler.step(val_loss)

            # 模型检查点和早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_checkpoint("best_model.pth")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {early_stopping_patience} epochs without improvement."
                )
                break

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

        print("Training finished.")
        print(f"Best validation loss: {best_val_loss:.4f}")

    def evaluate(self, loader: DataLoader):
        """
        在给定的数据集上评估模型。
        """
        loss, _ = self._run_epoch(loader, is_train=False)
        print(f"Evaluation Loss: {loss:.4f}")
        return loss

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        使用模型进行预测。
        """
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
        return outputs

    def predict_dataset(self, dataset: Dataset, return_targets: bool = False):
        """对整个数据集进行预测"""
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_data in loader:
                if len(batch_data) == 3:  # With mask
                    features, targets, mask = batch_data
                    features = features.to(self.device)
                    mask = mask.to(self.device)
                    outputs = self.model(features, src_padding_mask=mask)
                else:
                    features, targets = batch_data
                    features = features.to(self.device)
                    outputs = self.model(features)
                
                all_predictions.append(outputs.cpu())
                if return_targets:
                    all_targets.append(targets)
        
        predictions = torch.cat(all_predictions, dim=0)
        
        if return_targets:
            targets = torch.cat(all_targets, dim=0)
            return predictions, targets
        return predictions

    def save_checkpoint(self, filename: str):
        """
        保存模型检查点。
        """
        path = os.path.join(self.log_dir, filename)
        print(f"Saving checkpoint to {path}")
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename: str):
        """
        加载模型检查点。
        """
        path = os.path.join(self.log_dir, filename)
        if not os.path.exists(path):
            print(f"Checkpoint file not found: {path}")
            return
        print(f"Loading checkpoint from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def _calculate_metrics(self, outputs, targets):
        """计算多种评估指标"""
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()

        mse = F.mse_loss(outputs, targets).item()
        mae = mean_absolute_error(targets_np, outputs_np)
        r2 = r2_score(targets_np, outputs_np)

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': mse ** 0.5
        }

    def plot_training_history(self):
        """绘制训练曲线"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Training Loss')
        
        plt.subplot(1, 2, 2)
        # 添加其他指标的绘制
        plt.show()
