import itertools
import json
from typing import Optional, List
import torch
from torch_geometric.data import Data

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
    数据集管理器 (GNN 版本)。
    核心职责：加载、管理电解液配方数据，并将其转换为可供GNN使用的图数据对象。
    """

    def __init__(self, dataset_file: Optional[str] = None, build_features: bool = True):
        """
        初始化 Dataset 类。

        Args:
            dataset_file (str, optional): 配方 JSON 文件路径。
            build_features (bool): 是否在初始化时立即构建图数据。
        """
        self.library: MaterialLibrary = MLibrary
        self.formulas: list[Electrolyte] = []
        self.feature_dim = 178  # 单个节点（材料）的特征维度

        # --- GNN改造: 核心存储从Tensor列表变为图Data对象列表 ---
        self.graph_data: List[Data] = []
        # ---------------------------------------------------------

        self.electrolyte_counter = itertools.count(1)

        if dataset_file:
            self.from_json(dataset_file)
        if build_features and self.formulas:
            self._build_graphs()

    # ------------------------ 魔术方法 ------------------------ #
    def __len__(self) -> int:
        """返回数据集的样本总数 (图的数量)"""
        return len(self.graph_data)

    def __getitem__(self, idx: int) -> Data:
        """
        使数据集支持下标访问，返回一个图数据对象。
        这是 PyG DataLoader 工作所必需的。
        """
        if not self.graph_data:
            raise RuntimeError("图数据集尚未构建。")
        return self.graph_data[idx]

    # ------------------------ 实例方法 ------------------------ #

    def _build_graphs(self):
        """
        核心方法：从 self.formulas 生成并保存图数据对象列表。
        """
        self.graph_data = self.get_training_data()
        # 可以在此添加对图数据的验证
        if not self.graph_data:
            warnings.warn("未生成任何图数据，请检查输入文件和配方。")

    def _update_graphs(self):
        """更新图数据列表（增删配方后）"""
        if self.formulas:
            self._build_graphs()
        else:
            self.graph_data = []

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
            self._update_graphs()

    def del_formula(self, id: str, refresh: bool = True):
        """删除指定 ID 的配方"""
        self.formulas = [f for f in self.formulas if f.id != id]
        if refresh:
            self._update_graphs()

    def update_performance(self, formulas: list[Electrolyte], predictions: list[float]):
        """更新配方的性能数据（例如预测结果）"""
        for formula, pred in zip(formulas, predictions):
            formula.performance["conductivity"] = pred
        self._update_graphs()

    def get_training_data(self) -> List[Data]:
        """
        生成GNN训练数据。
        每个配方被转换成一个全连接的图。
        """
        graph_list: List[Data] = []
        for formula in self.formulas:
            if (
                "conductivity" in formula.performance
                and formula.performance["conductivity"] is not None
            ):
                # 1. 节点特征 (Node Features)
                node_features = formula.get_feature_matrix()
                if not node_features:
                    continue  # 跳过没有材料的无效配方

                x = torch.tensor(node_features, dtype=torch.float32)
                num_nodes = len(node_features)

                # 2. 边索引 (Edge Index) - 构建全连接图
                # 对于N个节点，创建所有可能的自环之外的边
                if num_nodes > 1:
                    edge_index = torch.tensor(
                        list(itertools.permutations(range(num_nodes), 2)),
                        dtype=torch.long,
                    ).t().contiguous()
                else:
                    # 如果只有一个节点，没有边
                    edge_index = torch.empty((2, 0), dtype=torch.long)

                # 3. 图标签 (Graph Label)
                y = torch.tensor(
                    [formula.performance["conductivity"]], dtype=torch.float32
                )

                # 4. 创建并存储图数据对象
                graph_list.append(Data(x=x, edge_index=edge_index, y=y))

        return graph_list

    def get_candidate_features(self) -> tuple[list[Electrolyte], list[Data]]:
        """
        获取数据集中所有待预测的候选配方及其图特征。
        此方法专为 Predictor 提供数据。
        """
        candidates = [
            f
            for f in self.formulas
            if "conductivity" not in f.performance
            or f.performance["conductivity"] is None
        ]
        
        graph_candidates: List[Data] = []
        for c in candidates:
            node_features = c.get_feature_matrix()
            if not node_features:
                continue

            x = torch.tensor(node_features, dtype=torch.float32)
            num_nodes = len(node_features)

            if num_nodes > 1:
                edge_index = torch.tensor(
                    list(itertools.permutations(range(num_nodes), 2)),
                    dtype=torch.long,
                ).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # 对于待预测的样本，y可以省略或设为-1
            graph_candidates.append(Data(x=x, edge_index=edge_index))

        return candidates, graph_candidates
