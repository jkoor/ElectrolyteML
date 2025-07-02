# elml/ml/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Optional

from .models import BaseModel


class Trainer:
    """
    模型训练总监。
    接收一个模型和准备好的数据加载器，负责执行完整的训练和验证流程。
    """

    def __init__(self, model: BaseModel, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Trainer将在 {self.device} 设备上运行。")

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 0.001,
        patience: int = 10,
        save_path: Optional[str] = "best_model.pth",
        warmup_epochs: int = 10,
        lr_patience: int = 10,  # <-- 新增: LR调度器的耐心
        lr_factor: float = 0.5,  # <-- 新增: LR降低的因子
    ) -> Dict[str, List[float]]:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        # 主调度器，使用可配置的参数
        main_scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=lr_factor, patience=lr_patience
        )

        # 预热调度器
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs
            if epoch < warmup_epochs
            else 1,
        )

        # 定义损失函数
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        print("--- 开始训练 ---")
        for epoch in range(epochs):
            # 训练循环
            self.model.train()
            epoch_train_loss = 0.0
            for batch in train_loader:  # GNN Dataloader 返回的是一个 Batch 对象
                # ... (内部逻辑不变)
                batch = batch.to(self.device)
                optimizer.zero_grad()
                # 模型现在直接接收 batch 对象
                y_pred = self.model(batch)
                loss = criterion(y_pred.squeeze(-1), batch.y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # 验证循环
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    # ... (内部逻辑不变)
                    batch = batch.to(self.device)
                    # 模型现在直接接收 batch 对象
                    y_pred = self.model(batch)
                    loss = criterion(y_pred.squeeze(-1), batch.y)
                    epoch_val_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            # --- 调度器更新逻辑 ---
            # 在预热阶段，只调用预热调度器
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                # 预热结束后，由主调度器接管
                main_scheduler.step(avg_val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # 早停逻辑
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
