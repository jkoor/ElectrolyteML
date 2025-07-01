# elml/ml/evaluator.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

from .models import BaseModel
from ..dataset import ElectrolyteDataset


class Evaluator:
    """
    模型评估器。
    负责加载训练好的模型，在测试数据集上进行评估，并计算关键性能指标。
    """

    def __init__(self, model: BaseModel, model_path: str, device: Optional[str] = None):
        """
        初始化 Evaluator。

        Args:
            model (BaseModel): 要评估的模型结构实例 (与训练时使用的模型一致)。
            model_path (str): 训练好的模型权重文件路径 (.pth)。
            device (Optional[str]): 运行设备 ('cuda' or 'cpu')。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        # 加载模型状态字典，并确保它在正确的设备上
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 确保模型处于评估模式
        print(f"Evaluator 已加载模型 '{model_path}'，将在 {self.device} 上运行评估。")

    def evaluate(
        self, test_loader: DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        在给定的测试数据加载器上执行模型评估。

        Args:
            test_loader (DataLoader): 包含测试数据的 DataLoader。

        Returns:
            Tuple[Dict[str, float], np.ndarray, np.ndarray]:
            一个元组，包含：
            - 评估指标 (MSE, RMSE, MAE, R-squared) 的字典。
            - 真实值的Numpy数组。
            - 预测值的Numpy数组。
        """
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # 创建掩码，与训练和预测时保持一致
                mask = (X_batch.sum(dim=-1) == 0).to(self.device)

                # 使用模型的 predict 方法进行预测
                y_pred = self.model.predict(X_batch, mask)

                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        # 将列表中的所有批次数据连接成一个大的 numpy 数组
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # 计算评估指标
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        rmse = np.sqrt(mse)

        metrics = {
            "MSE (均方误差)": mse,
            "RMSE (均方根误差)": rmse,
            "MAE (平均绝对误差)": mae,
            "R-squared (R²分数)": r2,
        }

        print("--- 模型评估结果 ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("--------------------")

        return metrics, all_targets, all_preds

    def plot_predictions(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        绘制真实值 vs 预测值的散点图，并修复中文显示问题。

        Args:
            targets (np.ndarray): 真实目标值。
            predictions (np.ndarray): 模型预测值。
            save_path (Optional[str]): 图片保存路径。如果提供，则保存图片。
        """
        # --- 新增代码：处理中文字体显示 ---
        try:
            # 优先尝试使用黑体，这是常见的无衬线中文字体
            plt.rcParams["font.sans-serif"] = ["SimHei"]
            # 解决负号'-'显示为方块的问题
            plt.rcParams["axes.unicode_minus"] = False
        except Exception as e:
            print(
                f"注意：未能设置中文字体'SimHei'，图像中的中文可能无法正常显示。错误: {e}"
            )
            print("您可以尝试安装 'SimHei' 字体或在代码中指定其他已安装的中文字体。")
        # --- 新增代码结束 ---

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(8, 8))

        plt.scatter(
            targets,
            predictions,
            alpha=0.7,
            edgecolors="k",
            s=50,
            label="Predictions vs. Actual Values",
        )

        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Situation (y=x)"
        )

        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title("Model Prediction Accuracy Evaluation", fontsize=14)
        plt.legend(fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Evaluation chart saved to: {save_path}")

        plt.show()
