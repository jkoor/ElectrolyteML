import itertools
import json
import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from ..battery import Electrolyte
from ..material import MaterialLibrary
from .. import MLibrary


class ElectrolyteDataset(Dataset):
    """
    电解液数据集管理器。
    核心职责：加载、管理电解液配方数据，提供数据访问接口。
    与具体机器学习模型解耦，专注于数据管理。
    """

    def __init__(
        self,
        dataset_file: Optional[str] = None,
        feature_mode: str = "sequence",
        max_components: int = 10,
        target_metric: str = "conductivity",
    ):
        """
        初始化 Dataset 类。

        Args:
            dataset_file (str, optional): 配方 JSON 文件路径。
            feature_mode (str): 特征向量计算方式，默认为 "sequence"。
            max_components (int): 数据集中可能出现的最大组分数量，默认为 10。
        """

        # PyTorch Dataset 的构造函数调用
        super().__init__()

        self.library: MaterialLibrary = MLibrary
        self.formulas: list[Electrolyte] = []
        self.electrolyte_counter = itertools.count(1)
        self.feature_mode = feature_mode
        self.target_metric = target_metric
        self.MAX_COMPONENTS = max_components
        self.FEATURE_DIM = 3 + 167 + 8  # 材料类型(3), 分子指纹(167), 物化性质(8) = 178

        if dataset_file:
            self.from_json(dataset_file)

    # ------------------------ 魔术方法 ------------------------ #
    def __len__(self) -> int:
        """返回数据集的样本总数"""
        return len(self.formulas)

    # 在 ElectrolyteDataset 类内部
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使数据集支持下标访问。
        为模型训练提供一个样本，包括特征和目标。
        """
        # 1. 根据索引获取原始的电解液配方对象
        electrolyte: Electrolyte = self.formulas[idx]

        # 2. 调用内部方法，动态生成该配方的特征张量
        feature_tensor = self._generate_feature_tensor(electrolyte)

        # 3. 获取目标值（例如电导率），并转换为张量
        target_tensor = self._generate_target_tensor(electrolyte)

        # 获取温度信息
        temperature = torch.tensor(
            [electrolyte.performance["temperature"]], dtype=torch.float32
        )

        return feature_tensor, temperature, target_tensor

    def __iter__(self):
        """支持迭代访问"""
        return iter(self.formulas)

    # ------------------------ 数据管理方法 ------------------------ #

    def from_json(self, file_path: str):
        """从 JSON 文件加载配方数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            formulas_data: list[dict] = json.load(f)

        for data in formulas_data:
            self.add_formula(Electrolyte.from_dict(data))

    def to_json(self, file_path: str):
        """保存配方数据到 JSON 文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([f.to_dict() for f in self.formulas], f, indent=4)

    def add_formula(self, electrolyte: Electrolyte):
        """添加新配方"""
        electrolyte._data.id = f"E{next(self.electrolyte_counter)}"
        self.formulas.append(electrolyte)

    def remove_formula(self, formula_id: str):
        """删除指定 ID 的配方"""
        self.formulas = [f for f in self.formulas if f.id != formula_id]

    def get_formula_by_id(self, formula_id: str) -> Optional[Electrolyte]:
        """根据ID获取配方"""
        for formula in self.formulas:
            if formula.id == formula_id:
                return formula
        return None

    def update_performance(self, formula_id: str, performance_data: dict):
        """更新指定配方的性能数据"""
        formula = self.get_formula_by_id(formula_id)
        if formula:
            formula.performance.update(performance_data)

    # ------------------------ 数据查询方法 ------------------------ #

    def get_formulas_with_performance(
        self, metric: str = "conductivity"
    ) -> list[Electrolyte]:
        """
        获取包含指定性能指标的配方列表。
        """
        return [
            f
            for f in self.formulas
            if metric in f.performance and f.performance[metric] is not None
        ]

    def get_formulas_without_performance(
        self, metric: str = "conductivity"
    ) -> list[Electrolyte]:
        """
        获取不包含指定性能指标的配方列表（待预测样本）。
        """
        return [
            f
            for f in self.formulas
            if metric not in f.performance or f.performance[metric] is None
        ]

    def filter_formulas(self, filter_func) -> list[Electrolyte]:
        """
        根据自定义过滤函数筛选配方。

        Args:
            filter_func: 接受 Electrolyte 对象并返回 bool 的函数
        """
        return [f for f in self.formulas if filter_func(f)]

    def get_performance_statistics(self, metric: str = "conductivity") -> dict:
        """
        获取指定性能指标的统计信息。
        """
        values = [
            f.performance[metric]
            for f in self.formulas
            if metric in f.performance and f.performance[metric] is not None
        ]

        if not values:
            return {"count": 0}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    def clear(self):
        """清空所有配方数据"""
        self.formulas.clear()
        self.electrolyte_counter = itertools.count(1)

    def copy(self) -> "ElectrolyteDataset":
        """创建数据集的深拷贝"""
        new_dataset = ElectrolyteDataset()
        new_dataset.formulas = [f for f in self.formulas]  # 浅拷贝配方列表
        return new_dataset

    # ------------------------ 特征提取方法 ------------------------ #
    def _generate_feature_tensor(self, electrolyte: Electrolyte) -> torch.Tensor:
        """
        根据设定的模式，为单个电解液对象生成特征张量。
        这是所有特征工程逻辑的集合点。
        """
        if self.feature_mode == "sequence":
            # 调用为Transformer等序列模型设计的特征生成函数
            return self._generate_sequence_features(electrolyte)

        elif self.feature_mode == "weighted_average":
            # 调用为MLP, XGBoost等模型设计的特征生成函数
            return self._generate_average_features(electrolyte)

        else:
            raise ValueError(f"未知的特征模式: {self.feature_mode}")

    def _generate_sequence_features(self, electrolyte: Electrolyte) -> torch.Tensor:
        """
        为序列模型（如Transformer）生成特征张量。
        输出形状为 (MAX_COMPONENTS, FEATURE_DIM+1)
        分子指纹(167), 材料类型(3), 物化性质(8), 组分占比(1) = 179
        [[分子指纹(167), 材料类型(3), 物化性质(8), 组分占比(1)], ...]
        """
        # 1. 初始化一个全零的张量用于填充
        sequence_tensor = torch.zeros(
            self.MAX_COMPONENTS, self.FEATURE_DIM + 1, dtype=torch.float32
        )

        # 添加锂盐
        for i, material in enumerate(
            electrolyte.salts + electrolyte.solvents + electrolyte.additives
        ):
            if i >= self.MAX_COMPONENTS:
                print(
                    f"警告：配方 {electrolyte.id} 的组分数量超过MAX_COMPONENTS，部分组分将被忽略。"
                )
                break

            standardized_features_tensor = torch.tensor(
                material.standardized_feature_values, dtype=torch.float32
            )  # 分子指纹(167) + 材料类型(3) + 物化性质(8)
            fraction_tensor = torch.tensor(
                [electrolyte.proportions[i] * 0.01], dtype=torch.float32
            )  # 组分占比(1)

            # 将材料特征和其占比信息拼接在一起
            # 分子指纹(167), 材料类型(3), 物化性质(8), 组分占比(1)
            final_component_vector = torch.cat(
                [
                    standardized_features_tensor,
                    fraction_tensor,
                ]
            )

            # 3d. 将最终的组分向量填入序列张量
            sequence_tensor[i] = final_component_vector

        return sequence_tensor

    def _generate_average_features(self, electrolyte: Electrolyte) -> torch.Tensor:
        """
        通过加权平均法生成一个固定长度的特征向量。
        输出形状为 (FEATURE_DIM,)
        """
        # 初始化加权平均特征向量
        weighted_features = torch.zeros(self.FEATURE_DIM, dtype=torch.float32)

        # 获取所有组分和对应的摩尔分数
        all_materials = electrolyte.salts + electrolyte.solvents + electrolyte.additives

        for i, material in enumerate(all_materials):
            # 获取组分的标准化特征
            material_features = torch.tensor(
                material.standardized_feature_values, dtype=torch.float32
            )

            # 获取组分的摩尔分数（转换为小数）
            weight = electrolyte.proportions[i] * 0.01

            # 加权累加到总特征向量
            weighted_features += material_features * weight

        return weighted_features

    def _generate_target_tensor(self, electrolyte: Electrolyte) -> torch.Tensor:
        """
        为单个电解液对象生成目标张量。
        """
        target_value = electrolyte.performance.get(self.target_metric, 0.0)
        # 对目标值进行 log1p 变换：log(1 + σ)
        # 这样可以避免电导率为0时的数学错误，同时保持数值稳定性
        target_value = torch.log1p(torch.tensor(target_value, dtype=torch.float32))
        return target_value

    @staticmethod
    def create_splits(
        dataset_file: str,
        feature_mode: str = "sequence",
        target_metric: str = "conductivity",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        random_seed: int = 42,
    ) -> tuple["ElectrolyteDataset", "ElectrolyteDataset", "ElectrolyteDataset"]:
        """
        从一个JSON文件创建并返回训练、验证、测试三个数据集实例。

        该方法会自动筛选出包含有效性能标签的样本进行划分。

        Args:
            dataset_file (str): 包含所有配方数据的JSON文件路径。
            feature_mode (str): 为所有新创建的数据集设置的特征模式。
            target_metric (str): 为所有新创建的数据集设置的目标性能指标。
            train_frac (float): 训练集所占比例。
            val_frac (float): 验证集所占比例。测试集比例将自动计算。
            random_seed (int): 用于复现随机打乱结果的种子。

        Returns:
            tuple: 包含训练集、验证集、测试集的三个 ElectrolyteDataset 实例。
        """

        # 1. 创建一个临时数据集实例以加载和筛选数据
        temp_dataset = ElectrolyteDataset(dataset_file=dataset_file)

        # 2. 筛选出所有包含有效性能标签的样本，这是我们划分的基础
        labeled_formulas = temp_dataset.get_formulas_with_performance(
            metric=target_metric
        )

        if not labeled_formulas:
            raise ValueError(
                f"数据文件 '{dataset_file}' 中没有找到任何包含性能指标 '{target_metric}' 的有效样本。"
            )

        # 3. 使用随机种子来保证每次划分的结果都一样
        random.seed(random_seed)
        random.shuffle(labeled_formulas)

        # 4. 计算切分点
        n_total = len(labeled_formulas)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)

        # 5. 切分数据列表
        train_formulas = labeled_formulas[:n_train]
        val_formulas = labeled_formulas[n_train : n_train + n_val]
        test_formulas = labeled_formulas[n_train + n_val :]

        # 6. 创建三个新的、配置正确的数据集实例
        train_dataset = ElectrolyteDataset(
            feature_mode=feature_mode, target_metric=target_metric
        )
        val_dataset = ElectrolyteDataset(
            feature_mode=feature_mode, target_metric=target_metric
        )
        test_dataset = ElectrolyteDataset(
            feature_mode=feature_mode, target_metric=target_metric
        )

        # 7. 将切分好的配方列表分别赋给新的实例
        train_dataset.formulas = train_formulas
        val_dataset.formulas = val_formulas
        test_dataset.formulas = test_formulas

        print(f"数据划分完成 (随机种子={random_seed}):")
        print(f"  - 训练集样本数: {len(train_dataset)}")
        print(f"  - 验证集样本数: {len(val_dataset)}")
        print(f"  - 测试集样本数: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset
