import itertools
import json
from typing import List, Tuple, Optional

from battery import Electrolyte
from material import MLibrary, MaterialLibrary


class ElectrolyteDataset:
    def __init__(self, dataset_file: Optional[str] = None):
        """
        初始化 Dataset 类。

        Args:
            dataset_file (str, optional): 配方 JSON 文件路径。
        """

        self.library: MaterialLibrary = MLibrary  # 加载材料库
        self.formulas: List[Electrolyte] = []  # 存储所有配方
        self.max_seq_len = 5
        self.feature_dim = (
            178  # 3 (type) + 7 (max attrs) + 167 (fingerprint) + 1 (proportion)
        )
        # 计数器
        self.electrolyte_counter = itertools.count(1)  # 电解液 ID 计数器

        if dataset_file:
            self.from_json(dataset_file)

    def from_json(self, file_path: str):
        """从 JSON 文件加载配方数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            formulas_data: list[dict] = json.load(f)

        for data in formulas_data:
            self.add_formula(Electrolyte.from_dict(data))

    def add_formula(self, electrolyte: Electrolyte):
        """添加新配方"""
        electrolyte._data.id = f"E{next(self.electrolyte_counter)}"
        self.formulas.append(electrolyte)

    def del_formula(self, id: str):
        """删除指定 ID 的配方"""
        self.formulas = [f for f in self.formulas if f.id != id]

    def to_json(self, file_path: str):
        """保存配方数据到 JSON 文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([f.to_dict() for f in self.formulas], f, indent=4)

    def get_training_data(self) -> Tuple[List[List[float]], List[float]]:
        """
        生成训练数据。

        Returns:
            Tuple[List[List[float]], List[float]]: 特征矩阵 X 和标签 y（电导率）。
        """
        X: list[list[float]] = []
        y: list[float] = []
        for formula in self.formulas:
            if (
                "conductivity" in formula.performance
                and formula.performance["conductivity"] is not None
            ):
                X.append(formula.get_feature_matrix())
                y.append(formula.performance["conductivity"])
        return X, y

    def get_candidates(self) -> Tuple[List[Electrolyte], List[List[float]]]:
        """
        获取候选配方及其特征。

        Returns:
            Tuple[List[Electrolyte], List[List[float]]]: 候选配方列表和特征矩阵。
        """
        candidates = [
            f
            for f in self.formulas
            if "conductivity" not in f.performance
            or f.performance["conductivity"] is None
        ]
        X_candidates = [c.get_features() for c in candidates]
        return candidates, X_candidates

    def update_performance(self, formulas: List[Electrolyte], predictions: List[float]):
        """更新配方的性能数据（例如预测结果）"""
        for formula, pred in zip(formulas, predictions):
            formula.performance["conductivity"] = pred

    def __len__(self) -> int:
        """返回配方总数"""
        return len(self.formulas)
