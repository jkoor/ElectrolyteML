from material import LithiumSalt, Solvent, Additive
from models import (
    LithiumSaltModel,
    SolventModel,
    AdditiveModel,
    ElectrolyteModel,
    AnodeModel,
    CathodeModel,
    BatteryModel,
    Component,
)


# 电解液类
class Electrolyte:
    __slots__: list[str] = [
        "_data",
        "_lithium_salts",
        "_solvents",
        "_additives",
    ]

    # 类变量，用于保存所有实例
    _instances: list["Electrolyte"] = []

    def __init__(
        self,
        data: ElectrolyteModel,
    ):
        """
        电解液配方类
        """

        self._data = data
        self._lithium_salts: list[LithiumSalt] = [
            LithiumSalt(item.material) for item in data.lithium_salts
        ]
        self._solvents: list[Solvent] = [
            Solvent(item.material) for item in data.solvents
        ]
        self._additives: list[Additive] = [
            Additive(item.material) for item in data.additives
        ]
        # 将当前实例添加到类变量 instances 中
        self._instances.append(self)

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
    # 1. 获取电解液配方中的锂盐
    @property
    def lithium_salts(self) -> list[LithiumSalt]:
        return self._lithium_salts

    # 2. 获取电解液配方中的溶剂
    @property
    def solvents(self) -> list[Solvent]:
        return self._solvents

    # 3. 获取电解液配方中的添加剂
    @property
    def additives(self) -> list[Additive]:
        return self._additives

    # ------------------------ 类方法 ------------------------ #

    # 1. 获取所有的 Electrolyte 实例
    @classmethod
    def all(cls) -> list["Electrolyte"]:
        """
        获取所有的 Electrolyte 实例
        """
        return cls._instances

    # 2. 查找 Electrolyte 实例
    @classmethod
    def find(cls, name=None, id=None):
        """
        查找 Material 实例
        """

        if not any([name, id]):
            raise ValueError("At least one search parameter must be provided.")

        for instance in cls._instances:
            if (name and instance.name == name) or (id and instance.id == id):
                return instance

        raise ValueError("Electrolyte not found.")

    # 3. 从json文件种读取电解液数据
    @classmethod
    def from_json(cls, json_file: str) -> "Electrolyte":
        """从 JSON 创建实例"""
        with open(json_file, "r", encoding="utf-8") as f:
            data = ElectrolyteModel.model_validate_json(f.read())
        return cls(data)

    # 4. 从dict字典读取电解液数据
    @classmethod
    def from_dict(cls, dict_data: dict) -> "Electrolyte":
        """从 dict 创建实例"""
        data = ElectrolyteModel.model_validate(dict_data)
        return cls(data)

    # 4. 使用数据模型创建电解液实例
    @classmethod
    def create(
        cls,
        name: str,
        id: str,
        description: str,
        lithium_salts: list[tuple[LithiumSaltModel, float]],
        solvents: list[tuple[SolventModel, float]],
        additives: list[tuple[AdditiveModel, float]],
        performance: dict,
    ) -> "Electrolyte":
        """使用数据模型创建电解液实例"""

        lithium_salts_component: list[Component[LithiumSaltModel]] = [
            (Component(material=salt[0], overall_fraction=salt[1]))
            for salt in lithium_salts
        ]
        solvents_component: list[Component[SolventModel]] = [
            (Component(material=solvent[0], relative_fraction=solvent[1]))
            for solvent in solvents
        ]
        additives_component: list[Component[AdditiveModel]] = [
            (Component(material=additive[0], overall_fraction=additive[1]))
            for additive in additives
        ]

        data = ElectrolyteModel(
            name=name,
            id=id,
            description=description,
            lithium_salts=lithium_salts_component,
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
            [
                f"{salt.material.abbreviation} ({salt.overall_fraction})"
                for salt in self._data.lithium_salts
            ]
        )
        solvent_str = ", ".join(
            [
                f"{solvent.material.abbreviation} ({solvent.relative_fraction}) ({solvent.overall_fraction})"
                for solvent in self._data.solvents
            ]
        )
        additive_str = ", ".join(
            [
                f"{additive.material.abbreviation} ({additive.overall_fraction})"
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
        return self._data.model_dump()

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


# 正极类
class Cathode:
    __slots__: list[str] = [
        "_data",
        "_active_materials",
        "_conductors",
        "_binders",
    ]

    # 类变量，用于保存所有实例

    def __init__(self, data: CathodeModel) -> None:
        self._data = data


# 负极类
class Anode:
    __slots__: list[str] = [
        "_data",
        "_active_materials",
        "_conductors",
        "_binders",
    ]

    # 类变量，用于保存所有实例

    def __init__(self, data: AnodeModel) -> None:
        self._data = data


# 电池类
class Battery:
    __slots__: list[str] = [
        "_data",
        "_anode",
        "_cathode",
        "_electrolyte",
    ]

    # 类变量，用于保存所有实例
    _instances: list["Battery"] = []

    def __init__(
        self,
        data: BatteryModel,
    ):
        """
        电池类
        """

        self._data: BatteryModel = data
        self._anode = Anode(data.anode)
        self._cathode = Cathode(data.cathode)
        self._electrolyte = Electrolyte(data.electrolyte)
        # 将当前实例添加到类变量 instances 中
        self._instances.append(self)

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
    # 1. 获取电池正极
    @property
    def cathode(self) -> Cathode:
        return self._cathode

    # 2. 获取电池负极
    @property
    def anode(self) -> Anode:
        return self._anode

    # 3. 获取电池电解液
    @property
    def electrolyte(self) -> Electrolyte:
        return self._electrolyte

    # ------------------------ 类方法 ------------------------ #

    # 1. 获取所有的 Battery 实例
    @classmethod
    def all(cls) -> list["Battery"]:
        """
        获取所有的 Battery 实例
        """
        return cls._instances

    # 2. 查找 Battery 实例
