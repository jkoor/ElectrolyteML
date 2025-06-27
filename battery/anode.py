from battery.models import AnodeModel


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
        self._data: AnodeModel = data
