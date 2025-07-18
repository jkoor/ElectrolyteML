from .models import BatteryModel
from .anode import Anode
from .cathode import Cathode
from .electrolyte import Electrolyte


# 电池类
class Battery:
    __slots__ = [
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
