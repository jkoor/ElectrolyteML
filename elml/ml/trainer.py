import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional

from .models import BaseModel
from ..dataset.electrolyte_dataset import ElectrolyteDataset


class Trainer:
    """
    模型训练总监。
    接收一个模型和数据集，负责执行完整的训练和验证流程。
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: ElectrolyteDataset,
        device: Optional[str] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Trainer将在 {self.device} 设备上运行。")

    def run(
        self,
        train_ratio: float = 0.8,
        batch_size: int = 32,
        epochs: int = 100,
        lr: float = 0.001,
        patience: int = 10,
        save_path: Optional[str] = "best_model.pth",
    ) -> Dict[str, List[float]]:
        # 1. 准备数据
        full_tensor_dataset = TensorDataset(self.dataset.features, self.dataset.targets)
        train_size = int(train_ratio * len(full_tensor_dataset))
        val_size = len(full_tensor_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_tensor_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 2. 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # 当 'val_loss' 在 patience=5 个周期内不下降时，学习率乘以 factor=0.1
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=5)

        # 3. 训练状态初始化
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        print("--- 开始训练 ---")
        for epoch in range(epochs):
            # 训练循环
            self.model.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                mask = (X_batch.sum(dim=-1) == 0).to(self.device)
                y_pred = self.model(X_batch, mask)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # 验证循环
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    mask = (X_batch.sum(dim=-1) == 0).to(self.device)
                    y_pred = self.model(X_batch, mask)
                    loss = criterion(y_pred, y_batch)
                    epoch_val_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            # 更新学习率调度器
            scheduler.step(avg_val_loss)
            # 使用 optimizer.param_groups 来获取当前的学习率
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}"
            )

            # 4. 早停与模型保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发：验证损失在 {patience} 个周期内未改善。")
                    break
        print("--- 训练结束 ---")
        return history
