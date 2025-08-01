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
    def condition(self) -> dict:
        return self._data.condition

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
        salts: list[tuple["Material", float]],
        solvents: list[tuple["Material", float]],
        additives: list[tuple["Material", float]],
        performance: dict[str, float],
        condition: dict[str, float] = {"temperature": 298.15},
    ) -> "Electrolyte":
        """
        使用数据模型创建电解液实例
        Args:
            name (str): 电解液名称
            id (str): 电解液ID
            description (str): 电解液描述
            salts (list[tuple[Material, float]]): 锂盐列表，每个元组包含锂盐对象和其占比
            solvents (list[tuple[Material, float]]): 溶剂列表，每个元组包含溶剂对象和其占比
            additives (list[tuple[Material, float]]): 添加剂列表，每个元组包含添加剂对象和其占比
            condition (dict): 测试条件 默认值为 {"temperature": 298.15}
            performance (dict): 性能参数
        """

        # 将锂盐材料转换为 Component 模型
        salts_component: list[Component] = []
        for salt, ratio in salts:
            salts_component_dict = {
                "abbr": salt.abbreviation,
                "cas_registry_number": salt.cas_registry_number,
                "overall_fraction": ratio,
            }
            salts_component.append(Component.model_validate(salts_component_dict))
        # 将溶剂材料转换为 Component 模型
        solvents_component: list[Component] = []
        for solvent, ratio in solvents:
            solvents_component_dict = {
                "abbr": solvent.abbreviation,
                "cas_registry_number": solvent.cas_registry_number,
                "relative_fraction": ratio,
            }
            solvents_component.append(Component.model_validate(solvents_component_dict))
        # 将添加剂材料转换为 Component 模型
        additives_component: list[Component] = []
        for additive, ratio in additives:
            additives_component_dict = {
                "abbr": additive.abbreviation,
                "cas_registry_number": additive.cas_registry_number,
                "overall_fraction": ratio,
            }
            additives_component.append(
                Component.model_validate(additives_component_dict)
            )
        # 创建 ElectrolyteModel 实例
        data = ElectrolyteModel(
            name=name,
            id=id,
            description=description,
            salts=salts_component,
            solvents=solvents_component,
            additives=additives_component,
            performance=performance,
            condition=condition,
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
