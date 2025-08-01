from functools import cached_property
from rdkit.Chem import MACCSkeys, Descriptors, MolFromSmiles, rdMolDescriptors  # type: ignore
from .models import MaterialModel


# 材料（Material）
class Material:
    __slots__: tuple[str, ...] = (
        "_data",
        "_mol",
        "_fingerprint",
        "_molecular_weight",
    )

    def __init__(self, data: MaterialModel):
        self._data = data
        self._mol = MolFromSmiles(self.molecular_structure)
        self._fingerprint = None  # 分子指纹
        self._molecular_weight = None  # 分子量

        # 检查分子结构是否正确
        if not self._mol:
            raise ValueError(
                f"{self._data.abbreviation} 材料分子结构 SMILES 表示错误。"
            )

    # ------------------------ 魔术方法 ------------------------ #
    # 1. 重写 __repr__ 方法，返回材料名称
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {self.abbreviation}, {self.cas_registry_number}, {self.molecular_structure})"

    # 2. 动态代理属性访问，直接访问 _data 的字段
    def __getattr__(self, name: str):
        """动态代理属性访问，直接访问 _data 的字段"""
        try:
            return getattr(self._data, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    # def __new__(cls, data: MaterialModel):
    #     # 如果材料已存在，直接返回池中的实例
    #     if data.cas_registry_number in cls._instances:
    #         return cls._instances[data.cas_registry_number]
    #     # 否则创建新实例并加入池中
    #     instance = super().__new__(cls)
    #     cls._instances[data.cas_registry_number] = instance
    #     return instance

    # ------------------------ 私有方法 ------------------------ #

    # ------------------------ 属性方法 ------------------------ #
    # 1. 从 _data 中获取属性
    @property
    def name(self) -> str:
        """返回材料名称"""
        return self._data.name

    @property
    def abbreviation(self) -> str:
        """返回材料缩写"""
        return self._data.abbreviation

    @property
    def cas_registry_number(self) -> str:
        """返回材料的 CAS 注册号"""
        return self._data.cas_registry_number

    @property
    def description(self) -> str:
        """返回材料描述"""
        return self._data.description

    @property
    def material_type(self) -> str:
        """返回材料类型"""
        return self._data.material_type.value

    @property
    def molecular_structure(self) -> str:
        """返回材料的分子结构 (SMILES 格式)"""
        return self._data.molecular_structure

    # 2. 计算分子指纹
    @cached_property
    def molecular_fingerprint(self) -> list[float]:
        """
        计算分子指纹
        """
        # mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        #     radius=2, fpSize=2048
        # )
        # # 使用Morgan指纹生成分子指纹（可以选择不同的指纹算法）
        # fingerprint = mfpgen.GetFingerprint(mol)

        # 使用MACCSkeys指纹生成分子指纹
        # 分子指纹长度166，为了编程方便（Python索引从0开始），实际上是一个167位的向量
        fingerprint = MACCSkeys._pyGenMACCSKeys(self._mol)

        # 将指纹转换为NumPy数组并转换为列表，供机器学习使用
        return list(fingerprint)

    # 3. 计算分子量
    @cached_property
    def molecular_weight(self):
        """
        计算分子量。
        """
        # return Descriptors.CalcMolDescriptors(mol).get("MolWt")
        return Descriptors.MolWt(self._mol)  # type: ignore

    # 4. 获取材料分子信息
    @property
    def molecular_descriptor(self) -> dict:
        """
        返回材料的分子信息。
        """

        return Descriptors.CalcMolDescriptors(self._mol)

    # 5. 获取材料分子式
    @property
    def molecular_formula(self) -> str:
        """
        返回材料的分子式。
        """
        mol = MolFromSmiles(self.molecular_structure)
        return rdMolDescriptors.CalcMolFormula(mol)

    # 6. 获取材料标准化的特征张量
    @cached_property
    def standardized_feature_values(self) -> list[float]:
        """
        返回材料的标准化特征张量。
        """
        from .. import MLibrary

        # 类型映射字典
        _material_type_vector_map: dict[str, list[float]] = {
            "Salt": [1, 0, 0],
            "Solvent": [0, 1, 0],
            "Additive": [0, 0, 1],
        }
        fingerprint: list[float] = self.molecular_fingerprint  # 分子指纹(167)
        type_list: list[float] = _material_type_vector_map[
            self.material_type
        ]  # 材料类型(3)
        standardized_features: list[float] = list(
            MLibrary.get_standardized_value(self).values()
        )  # 物化性质(8)

        # 合并特征
        # 分子指纹(167), 材料类型(3), 物化性质(8)
        features: list[float] = fingerprint + type_list + standardized_features
        return features

    # ------------------------ 类方法 ------------------------ #

    # 1. 使用数据模型创建材料实例
    @classmethod
    def create(
        cls,
        name,
        abbreviation,
        material_type,
        cas_registry_number,
        description,
        molecular_structure,
        density,
        melting_point,
        boiling_point,
    ) -> "Material":
        """使用数据模型创建材料实例"""

        data = MaterialModel(
            name=name,
            abbreviation=abbreviation,
            material_type=material_type,
            cas_registry_number=cas_registry_number,
            description=description,
            molecular_structure=molecular_structure,
            density=density,
            melting_point=melting_point,
            boiling_point=boiling_point,
        )
        return cls(data)

    # ------------------------ 实例方法 ------------------------ #
