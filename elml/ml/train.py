import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from typing import Optional, Union
import warnings

from sklearn.metrics import r2_score, mean_absolute_error

from .models import ElectrolyteMLP, ElectrolyteTransformer

# 过滤PyTorch嵌套张量的警告
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")


class ElectrolyteTrainer:
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
        num_workers: int = 0,
        log_dir: str = "runs",
        model_checkpoint_path: str = "best_model.pth",
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
            num_workers (int): DataLoader的工作线程数。
            log_dir (str): 保存日志和模型检查点的目录。
            model_checkpoint_path (str, optional): 预训练模型检查点路径。
            optimizer_name (str): 优化器名称 ("adamw", "adam" 或 "sgd")。
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_dir = log_dir
        self.model_checkpoint_path = model_checkpoint_path

        # 自动检测设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)

        # 初始化DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

        # 初始化优化器、损失函数和学习率调度器
        if optimizer_name == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            self.optimizer = Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
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
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

    def _run_epoch(self, loader: DataLoader, is_train: bool = True):
        """运行一个epoch的内部辅助函数。"""
        self.model.train() if is_train else self.model.eval()
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()

        for batch_data in loader:
            # 现在数据集返回 (features, temperature, targets)
            if len(batch_data) == 3:
                features, temperature, targets = batch_data
                features = features.to(self.device)
                temperature = temperature.to(self.device)
                targets = targets.to(self.device)

                # 判断模型类型并调用相应的forward方法
                if isinstance(self.model, ElectrolyteTransformer):
                    # 为Transformer模型创建padding mask
                    # 检查features中全零的行，这些是填充的位置
                    padding_mask = features.sum(dim=-1) == 0  # [batch_size, seq_len]
                    outputs = self.model(
                        features, temperature, src_padding_mask=padding_mask
                    )
                else:
                    # 对于MLP模型，temperature需要与features拼接
                    # 注意：这里假设MLP模型也需要温度信息，可能需要修改MLP模型
                    outputs = self.model(features)
            else:
                raise ValueError(
                    f"Expected 3 elements in batch_data, got {len(batch_data)}"
                )

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
                self.save_checkpoint(self.model_checkpoint_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {early_stopping_patience} epochs without improvement."
                )
                break

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

        print("Training finished.")
        print(f"Best validation loss: {best_val_loss:.4f}")
        return {
            "best_val_loss": best_val_loss,
            "history": self.history,
            "total_epochs": epoch + 1,
            "model_checkpoint_path": self.model_checkpoint_path,
        }

    def evaluate(self, loader: DataLoader):
        """
        在给定的数据集上评估模型。
        """
        loss, _ = self._run_epoch(loader, is_train=False)
        print(f"Evaluation Loss: {loss:.4f}")
        return loss

    def save_checkpoint(self, filepath: str):
        """
        保存模型检查点。
        """
        path = filepath
        print(f"Saving checkpoint to {path}")
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filepath: str):
        """
        加载模型检查点。
        """
        path = filepath
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

        return {"mse": mse, "mae": mae, "r2": r2, "rmse": mse**0.5}

    def plot_training_history(self):
        """绘制训练曲线"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Training Loss")

        plt.subplot(1, 2, 2)
        # 添加其他指标的绘制
        plt.show()
