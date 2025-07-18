from battery.models import CathodeModel


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
        self._data: CathodeModel = data
