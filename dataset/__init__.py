import itertools
import json
from typing import Tuple, Optional
from typing import List, Union, Dict, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from battery import Electrolyte
from material import MLibrary, MaterialLibrary
from ml import BaseModel, ElectrolyteTransformer, ElectrolyteMLP


class ElectrolyteDataset:
    # 模型注册表
    MODEL_REGISTRY = {"transformer": ElectrolyteTransformer, "mlp": ElectrolyteMLP}

    def __init__(self, dataset_file: Optional[str] = None, build_features: bool = True):
        """
        初始化 Dataset 类。

        Args:
            dataset_file (str, optional): 配方 JSON 文件路径。
        """

        self.library: MaterialLibrary = MLibrary  # 加载材料库
        self.formulas: list[Electrolyte] = []  # 存储所有配方

        self.max_seq_len = 5  # 配方材料最大数量
        self.feature_dim = (
            178  # 3 (type) + 7 (max attrs) + 167 (fingerprint) + 1 (proportion)
        )

        self.features: torch.Tensor = None  # type: ignore
        self.targets: torch.Tensor = None  # type: ignore

        # 计数器
        self.electrolyte_counter = itertools.count(1)  # 电解液 ID 计数器

        if dataset_file:
            self.from_json(dataset_file)
        if build_features and self.formulas:
            self._build_features()

    # ------------------------ 魔术方法 ------------------------ #
    def __len__(self) -> int:
        """返回配方总数"""
        return len(self.formulas)

    # ------------------------ 实例方法 ------------------------ #

    def _build_features(self):
        """生成并保存特征矩阵和目标值"""
        X, y = self.get_training_data()
        if X:
            self.features = torch.tensor(
                X, dtype=torch.float32
            )  # [num_formulas, max_seq_len, feature_dim]
            self.targets = torch.tensor(y, dtype=torch.float32).unsqueeze(
                1
            )  # [num_formulas, 1]
            # 验证数据
            if torch.isnan(self.features).any() or torch.isinf(self.features).any():
                raise ValueError("特征矩阵包含 NaN 或 Inf")
            if torch.isnan(self.targets).any() or torch.isinf(self.targets).any():
                raise ValueError("目标值包含 NaN 或 Inf")
        else:
            self.features = torch.tensor([], dtype=torch.float32)
            self.targets = torch.tensor([], dtype=torch.float32)

    def _update_features(self):
        """更新特征矩阵（增删配方后）"""
        if self.formulas:
            self._build_features()
        else:
            self.features = torch.tensor([], dtype=torch.float32)
            self.targets = torch.tensor([], dtype=torch.float32)

    def from_json(self, file_path: str):
        """从 JSON 文件加载配方数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            formulas_data: list[dict] = json.load(f)

        for data in formulas_data:
            self.add_formula(Electrolyte.from_dict(data), refresh=False)

    def to_json(self, file_path: str):
        """保存配方数据到 JSON 文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([f.to_dict() for f in self.formulas], f, indent=4)

    def add_formula(self, electrolyte: Electrolyte, refresh: bool = True):
        """添加新配方"""
        electrolyte._data.id = f"E{next(self.electrolyte_counter)}"
        self.formulas.append(electrolyte)
        if refresh:
            self._update_features()

    def del_formula(self, id: str, refresh: bool = True):
        """删除指定 ID 的配方"""
        self.formulas = [f for f in self.formulas if f.id != id]
        if refresh:
            self._update_features()

    def get_training_data(self) -> Tuple[list[list[list[float]]], list[float]]:
        """
        生成训练数据。

        Returns:
            Tuple[List[List[float]], List[float]]: 特征矩阵 X 和标签 y（电导率）。
        """
        X: list[list[list[float]]] = []
        y: list[float] = []
        for formula in self.formulas:
            if (
                "conductivity" in formula.performance
                and formula.performance["conductivity"] is not None
            ):
                formula_features = formula.get_feature_matrix()
                # 验证特征维度
                for feature in formula_features:
                    if len(feature) != self.feature_dim:
                        raise ValueError(
                            f"配方 {formula._data.id} 的特征维度错误："
                            f"预期 {self.feature_dim}，实际为 {len(feature)}"
                        )
                # 填充到 max_seq_len
                while len(formula_features) < self.max_seq_len:
                    formula_features.append([0.0] * self.feature_dim)
                # 截断超长序列
                formula_features = formula_features[: self.max_seq_len]
                # 验证填充后长度
                if len(formula_features) != self.max_seq_len:
                    raise ValueError(
                        f"配方 {formula._data.id} 填充后材料数量为 {len(formula_features)}，"
                        f"预期为 {self.max_seq_len}"
                    )
                X.append(formula_features)
                y.append(formula.performance["conductivity"])
        return X, y

    def update_performance(self, formulas: list[Electrolyte], predictions: list[float]):
        """更新配方的性能数据（例如预测结果）"""
        for formula, pred in zip(formulas, predictions):
            formula.performance["conductivity"] = pred
        self._update_features()

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        if self.features is None or self.targets is None:
            self._build_features()
        dataset = TensorDataset(self.features, self.targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_model(
        self,
        model: Union[str, BaseModel, Type[BaseModel]] = "transformer",
        model_config: Optional[Dict] = None,
        train_ratio: float = 0.8,
        batch_size: int = 32,
        epochs: int = 100,
        lr: float = 0.001,
        patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: Optional[str] = "best_model.pth",
    ) -> Dict:
        """
        训练模型并评估性能。

        Args:
            model: 模型类型（字符串，如 "transformer"）、模型类或实例。
            model_config: 模型配置字典。
            train_ratio: 训练集比例。
            batch_size: 批量大小。
            epochs: 训练轮数。
            lr: 学习率。
            patience: 早停耐心值。
            device: 设备（"cuda" 或 "cpu"）。
            save_path: 模型保存路径。

        Returns:
            Dict: 训练历史（损失和评估指标）。
        """
        model_config = model_config or {}
        if isinstance(model, str):
            if model not in self.MODEL_REGISTRY:
                raise ValueError(
                    f"不支持的模型类型：{model}，可用模型：{list(self.MODEL_REGISTRY.keys())}"
                )
            model_class = self.MODEL_REGISTRY[model]
            model_instance = model_class(input_dim=self.feature_dim, **model_config)
        elif isinstance(model, type) and issubclass(model, BaseModel):
            model_instance = model(input_dim=self.feature_dim, **model_config)
        elif isinstance(model, BaseModel):
            model_instance = model
        else:
            raise ValueError("model 参数必须是字符串、BaseModel 子类或实例")

        model_instance.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr)

        dataset = TensorDataset(self.features, self.targets)
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_r2": [],
            "val_mre": [],
        }
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self._train_loop(
                model_instance, train_loader, criterion, optimizer, device
            )
            val_metrics = self._evaluate(model_instance, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["mse"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_r2"].append(val_metrics["r2"])
            history["val_mre"].append(val_metrics["mre"])

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                f"Val MSE: {val_metrics['mse']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, Val R²: {val_metrics['r2']:.4f}, "
                f"Val MRE: {val_metrics['mre']:.4f}"
            )

            if val_metrics["mse"] < best_val_loss:
                best_val_loss = val_metrics["mse"]
                patience_counter = 0
                if save_path:
                    torch.save(model_instance.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停：验证损失在 {patience} 个 epoch 未改善")
                    break

        # 最终验证集评估
        print("\n最终验证集评估：")
        final_metrics = self.evaluate_model(model_instance, val_loader, device)
        print(
            f"MSE: {final_metrics['mse']:.4f}, RMSE: {final_metrics['rmse']:.4f}, "
            f"MAE: {final_metrics['mae']:.4f}, R²: {final_metrics['r2']:.4f}, "
            f"MRE: {final_metrics['mre']:.4f}"
        )

        return history

    def _train_loop(
        self, model: BaseModel, loader: DataLoader, criterion, optimizer, device: str
    ) -> float:
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            mask = (X_batch.sum(dim=-1) == 0).to(device)
            y_pred = model(X_batch, mask)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)

    def _evaluate(
        self, model: BaseModel, loader: DataLoader, criterion, device: str
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                mask = (X_batch.sum(dim=-1) == 0).to(device)
                y_pred = model(X_batch, mask)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                preds.append(y_pred.cpu().numpy())
                targets.append(y_batch.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()
        mse = total_loss / len(loader.dataset)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, preds)
        r2 = r2_score(targets, preds)
        # 计算平均相对误差（避免除以零）
        relative_errors = [
            abs(p - t) / max(abs(t), 1e-8) for p, t in zip(preds, targets)
        ]
        mre = np.mean(relative_errors)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mre": mre}

    def get_candidates(self) -> Tuple[List[Electrolyte], torch.Tensor]:
        candidates = [
            f
            for f in self.formulas
            if "conductivity" not in f.performance
            or f.performance["conductivity"] is None
        ]
        X_candidates = []
        for c in candidates:
            features = c.get_feature_matrix()
            for feature in features:
                if len(feature) != self.feature_dim:
                    raise ValueError(
                        f"候选配方 {c._data.id} 的特征维度错误："
                        f"预期 {self.feature_dim}，实际为 {len(feature)}"
                    )
            while len(features) < self.max_seq_len:
                features.append([0.0] * self.feature_dim)
            features = features[: self.max_seq_len]
            X_candidates.append(features)
        return candidates, torch.tensor(
            X_candidates, dtype=torch.float32
        ) if X_candidates else torch.tensor([]).reshape(
            0, self.max_seq_len, self.feature_dim
        )

    def evaluate_model(
        self, model: BaseModel, loader: DataLoader, device: str
    ) -> Dict[str, float]:
        """
        评估模型性能。

        Args:
            model: 已训练的模型。
            loader: 数据加载器（验证集或测试集）。
            device: 设备。

        Returns:
            Dict: 包含 MSE, RMSE, MAE, R², MRE 的评估指标。
        """
        criterion = nn.MSELoss()
        metrics = self._evaluate(model, loader, criterion, device)
        return metrics

    def predict_candidates(
        self,
        model: BaseModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> List[Tuple[Electrolyte, float]]:
        """
        预测候选配方的电导率。

        Args:
            model: 已训练的模型。
            device: 设备。

        Returns:
            List[Tuple[Electrolyte, float]]: 候选配方及其预测电导率。
        """
        candidates, X_candidates = self.get_candidates()
        if not candidates:
            return []
        model.eval()
        X_candidates = X_candidates.to(device)
        mask = (X_candidates.sum(dim=-1) == 0).to(device)
        with torch.no_grad():
            preds = model.predict(X_candidates, mask).cpu().numpy().flatten()
        return [(c, pred) for c, pred in zip(candidates, preds)]
