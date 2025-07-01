import itertools
import json
from typing import Tuple, Optional, List
import torch

from ..battery import Electrolyte
from ..material import MaterialLibrary
from .. import MLibrary


import warnings

# 忽略特定的PyTorch警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage*",
)


class ElectrolyteDataset:
    """
    数据集管理器。
    核心职责：加载、管理电解液配方数据，并将其转换为可供机器学习使用的特征和标签。
    """

    def __init__(self, dataset_file: Optional[str] = None, build_features: bool = True):
        """
        初始化 Dataset 类。

        Args:
            dataset_file (str, optional): 配方 JSON 文件路径。
            build_features (bool): 是否在初始化时立即构建特征矩阵。
        """
        self.library: MaterialLibrary = MLibrary
        self.formulas: list[Electrolyte] = []

        # 这些参数定义了数据的形状，对于特征工程至关重要
        self.max_seq_len = 5
        self.feature_dim = 178

        self.features: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None

        self.electrolyte_counter = itertools.count(1)

        if dataset_file:
            self.from_json(dataset_file)
        if build_features and self.formulas:
            self._build_features()

    # ------------------------ 魔术方法 ------------------------ #
    def __len__(self) -> int:
        """返回数据集的样本总数 (基于特征矩阵)"""
        if self.features is not None:
            return len(self.features)
        return 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使数据集支持下标访问。
        这是 DataLoader 工作所必需的。
        """
        if self.features is None or self.targets is None:
            raise RuntimeError("Dataset features/targets have not been built yet.")

        return self.features[idx], self.targets[idx]

    # ------------------------ 实例方法 ------------------------ #

    def _build_features(self):
        """
        核心方法：从 self.formulas 生成并保存特征矩阵和目标值。
        此方法保持不变。
        """
        X, y = self.get_training_data()
        if X:
            self.features = torch.tensor(X, dtype=torch.float32)
            self.targets = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            # 验证数据
            if torch.isnan(self.features).any() or torch.isinf(self.features).any():
                raise ValueError("特征矩阵包含 NaN 或 Inf")
            if torch.isnan(self.targets).any() or torch.isinf(self.targets).any():
                raise ValueError("目标值包含 NaN 或 Inf")
        else:
            self.features = torch.empty(0, self.max_seq_len, self.feature_dim)
            self.targets = torch.empty(0, 1)

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

    def update_performance(self, formulas: list[Electrolyte], predictions: list[float]):
        """更新配方的性能数据（例如预测结果）"""
        for formula, pred in zip(formulas, predictions):
            formula.performance["conductivity"] = pred
        self._update_features()

    def get_training_data(self) -> Tuple[list[list[list[float]]], list[float]]:
        """
        生成训练数据。
        """
        X: list[list[list[float]]] = []
        y: list[float] = []
        for formula in self.formulas:
            if (
                "conductivity" in formula.performance
                and formula.performance["conductivity"] is not None
            ):
                formula_features = formula.get_feature_matrix()  #

                # 验证每个材料的特征维度是否正确
                for feature in formula_features:
                    if len(feature) != self.feature_dim:
                        raise ValueError(
                            f"配方 {formula._data.id} 的特征维度错误："
                            f"预期 {self.feature_dim}，实际为 {len(feature)}"
                        )

                # --- 这是修复问题的关键代码块 ---
                # 1. 填充（Padding）：如果材料数量少于 max_seq_len，就用 0 向量填充
                while len(formula_features) < self.max_seq_len:
                    formula_features.append([0.0] * self.feature_dim)

                # 2. 截断（Truncation）：如果材料数量多于 max_seq_len，就截取前面的部分
                formula_features = formula_features[: self.max_seq_len]

                # 3. 最终验证：确保处理后的长度一定是 max_seq_len
                if len(formula_features) != self.max_seq_len:
                    # 这个错误理论上不应该发生，除非上面的逻辑有误
                    raise ValueError(
                        f"配方 {formula._data.id} 填充后材料数量为 {len(formula_features)}，"
                        f"预期为 {self.max_seq_len}"
                    )
                # --- 关键代码块结束 ---

                X.append(formula_features)
                y.append(formula.performance["conductivity"])

        return X, y

    def get_candidate_features(self) -> Tuple[List[Electrolyte], torch.Tensor]:
        """
        获取数据集中所有待预测的候选配方及其特征。
        此方法专为 Predictor 提供数据。
        """
        candidates = [
            f
            for f in self.formulas
            if "conductivity" not in f.performance
            or f.performance["conductivity"] is None
        ]
        X_candidates_list = []
        for c in candidates:
            features = c.get_feature_matrix()  #
            # ... (填充和维度检查逻辑与 get_training_data 类似)
            while len(features) < self.max_seq_len:
                features.append([0.0] * self.feature_dim)
            features = features[: self.max_seq_len]
            X_candidates_list.append(features)

        if not X_candidates_list:
            return [], torch.empty(0, self.max_seq_len, self.feature_dim)

        X_candidates_tensor = torch.tensor(X_candidates_list, dtype=torch.float32)
        return candidates, X_candidates_tensor
