from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 只在类型检查时导入，运行时不执行
    from ..material import (
        Material,
    )

from .. import MLibrary
from .models import ElectrolyteModel, Component


# 电解液类
class Electrolyte:
    __slots__ = ["_data", "_salts", "_solvents", "_additives", "proportions"]

    # 类变量，用于保存所有实例
    _instances: list["Electrolyte"] = []

    def __init__(
        self,
        data: ElectrolyteModel,
    ):
        """
        电解液配方类
        """

        self.proportions: list[float] = []
        self._data: ElectrolyteModel = data
        self._salts: list["Material"] = []
        self._solvents: list["Material"] = []
        self._additives: list["Material"] = []

        for item in data.salts:
            self._salts.append(
                MLibrary.get_material(item.abbr, item.cas_registry_number)
            )
            self.proportions.append(
                item.overall_fraction
            ) if item.overall_fraction else None

        for item in data.solvents:
            self._solvents.append(
                MLibrary.get_material(item.abbr, item.cas_registry_number)
            )
            self.proportions.append(
                item.overall_fraction
            ) if item.overall_fraction else None

        for item in data.additives:
            self._additives.append(
                MLibrary.get_material(item.abbr, item.cas_registry_number)
            )
            self.proportions.append(
                item.overall_fraction
            ) if item.overall_fraction else None

    # ------------------------ 魔术方法 ------------------------ #

    # 1. 重写 __repr__ 方法，返回材料名称
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {self.description})"

    # 2. 动态代理属性访问，直接访问 _data 的字段
    def __getattr__(self, name: str):
        """动态代理属性访问，直接访问 _data 的字段"""
        try:
            return getattr(self._data, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    # ------------------------ 私有方法 ------------------------ #

    # ------------------------ 属性方法 ------------------------ #
    # 1. 从_data 中获取属性
    @property
    def name(self) -> str:
        return self._data.name

    @property
    def description(self) -> str:
        return self._data.description

    @property
    def performance(self) -> dict:
        return self._data.performance

    # 2. 获取电解液配方中的锂盐
    @property
    def salts(self) -> list["Material"]:
        return self._salts

    # 3. 获取电解液配方中的溶剂
    @property
    def solvents(self) -> list["Material"]:
        return self._solvents

    # 4. 获取电解液配方中的添加剂
    @property
    def additives(self) -> list["Material"]:
        return self._additives

    # ------------------------ 类方法 ------------------------ #
    # 3. 从json文件种读取电解液数据
    @classmethod
    def from_json(cls, json_file: str) -> "Electrolyte":
        """从 JSON 创建实例"""
        with open(json_file, "r", encoding="utf-8") as f:
            data: ElectrolyteModel = ElectrolyteModel.model_validate_json(f.read())
        return cls(data)

    # 4. 从dict字典读取电解液数据
    @classmethod
    def from_dict(cls, dict_data: dict) -> "Electrolyte":
        """从 dict 创建实例"""
        data: ElectrolyteModel = ElectrolyteModel.model_validate(dict_data)
        return cls(data)

    # 6. 使用数据模型创建电解液实例
    @classmethod
    def create(
        cls,
        name: str,
        id: str,
        description: str,
        salts: list[dict],
        solvents: list[dict],
        additives: list[dict],
        performance: dict,
    ) -> "Electrolyte":
        """使用数据模型创建电解液实例"""

        salts_component: list[Component] = [
            Component.model_validate(salt) for salt in salts
        ]
        solvents_component: list[Component] = [
            (Component.model_validate(solvent)) for solvent in solvents
        ]
        additives_component: list[Component] = [
            (Component.model_validate(additive)) for additive in additives
        ]

        data = ElectrolyteModel(
            name=name,
            id=id,
            description=description,
            salts=salts_component,
            solvents=solvents_component,
            additives=additives_component,
            performance=performance,
        )

        return cls(data)

    # ------------------------ 实例方法 ------------------------ #

    # 1. 展示电解液配方
    def show(self):
        """展示电解液配方"""
        print(f"电解液配方: {self.name}")
        salt_str = ", ".join(
            [f"{salt.abbr} ({salt.overall_fraction})" for salt in self._data.salts]
        )
        solvent_str = ", ".join(
            [
                f"{solvent.abbr} ({solvent.relative_fraction}) ({solvent.overall_fraction})"
                for solvent in self._data.solvents
            ]
        )
        additive_str = ", ".join(
            [
                f"{additive.abbr} ({additive.overall_fraction})"
                for additive in self._data.additives
            ]
        )
        print(f"锂盐: {salt_str}")
        print(f"溶剂: {solvent_str}")
        print(f"添加剂: {additive_str}")

    # 2. 序列化为 JSON
    def to_json(self, json_file: str):
        """序列化为 JSON"""
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(self._data.model_dump_json(indent=4))

    # 3. 序列化为 dict
    def to_dict(self) -> dict:
        """序列化为 dict"""
        return self._data.model_dump(mode="json")

    # 4. 添加性能参数
    def set_performance(
        self,
        ionic_conductivity=None,
        viscosity=None,
        electrochemical_window=None,
        thermal_stability=None,
    ):
        if ionic_conductivity is not None:
            self.performance["ionic_conductivity"] = ionic_conductivity  # mS/cm
        if viscosity is not None:
            self.performance["viscosity"] = viscosity  # cP
        if electrochemical_window is not None:
            self.performance["electrochemical_window"] = electrochemical_window  # V
        if thermal_stability is not None:
            self.performance["thermal_stability"] = thermal_stability  # ℃

    # 5. 获取电解液配方的特征矩阵
    def get_feature_matrix(self) -> list[list[float]]:
        """返回每个材料的类型嵌入 + 特定属性 + 指纹 + 占比"""
        raw_features = []
        for i, material in enumerate(self.salts + self.solvents + self.additives):
            # 类型映射字典
            _material_type_vector_map: dict[str, list] = {
                "Salt": [1, 0, 0],
                "Solvent": [0, 1, 0],
                "Additive": [0, 0, 1],
            }
            type_embedding: list[float] = _material_type_vector_map[
                material.material_type
            ]
            attrs_dict: dict[str, float] = MLibrary.get_standardized_value(material)
            attrs_list: list[float] = list(attrs_dict.values())
            prop = self.proportions[i] * 0.01  # 占比转换为小数
            # 合并特征
            features: list[float] = (
                type_embedding + attrs_list + material.molecular_fingerprint + [prop]
            )
            raw_features.append(features)
        return raw_features
