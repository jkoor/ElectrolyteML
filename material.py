from functools import cached_property
import json
from typing import Generic, TypeVar, ClassVar
from rdkit.Chem import MACCSkeys, Descriptors, MolFromSmiles, rdMolDescriptors
import numpy as np

from models import (
    MaterialType,
    MaterialModel,
    LithiumSaltModel,
    SolventModel,
    AdditiveModel,
)

# 类型变量，用于泛型设计
T = TypeVar("T", MaterialModel, SolventModel, LithiumSaltModel, AdditiveModel)


# 父类：材料（Material）
class Material(Generic[T]):
    __slots__: tuple[str, ...] = (
        "_data",
        "_fingerprint",
        "_molecular_weight",
    )  # 只存储 Pydantic 数据对象
    # 类变量，用于保存所有实例
    _instances: ClassVar[list["Material"]] = []
    _data_model = MaterialModel  # 数据模型

    def __init__(self, data: MaterialModel):
        self._data: MaterialModel = data
        self._fingerprint = None  # 分子指纹
        self._molecular_weight = None  # 分子量
        # 将当前实例添加到类变量 instances 中
        self._instances.append(self)

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

    # ------------------------ 私有方法 ------------------------ #

    # ------------------------ 属性方法 ------------------------ #

    # 1. 计算分子指纹
    @cached_property
    def molecular_fingerprint(self):
        """
        计算分子指纹
        """
        # 使用RDKit生成分子对象
        mol = MolFromSmiles(self.molecular_structure)
        # mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        #     radius=2, fpSize=2048
        # )
        # # 使用Morgan指纹生成分子指纹（可以选择不同的指纹算法）
        # fingerprint = mfpgen.GetFingerprint(mol)

        # 使用MACCSkeys指纹生成分子指纹
        fingerprint = MACCSkeys._pyGenMACCSKeys(mol)

        # 将指纹转换为NumPy数组，供机器学习使用
        return np.array(fingerprint)

    # 2. 计算分子量
    @cached_property
    def molecular_weight(self):
        """
        计算分子量。
        """
        # 使用RDKit计算分子量
        mol = MolFromSmiles(self.molecular_structure)
        # return Descriptors.CalcMolDescriptors(mol).get("MolWt")
        return Descriptors.MolWt(mol)  # type: ignore

    # 3. 获取材料分子信息
    @property
    def molecular_descriptor(self) -> dict:
        """
        返回材料的分子信息。
        """

        mol = MolFromSmiles(self.molecular_structure)
        return Descriptors.CalcMolDescriptors(mol)

    # 4. 获取材料分子式
    @property
    def molecular_formula(self) -> str:
        """
        返回材料的分子式。
        """
        mol = MolFromSmiles(self.molecular_structure)
        return rdMolDescriptors.CalcMolFormula(mol)

    @property
    def instances(self):
        """返回所有实例"""
        return self._instances

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

    # 2. 从json文件种读取材料数据
    @classmethod
    def from_json(cls, json_file: str) -> "Material":
        """从 JSON 创建实例"""
        with open(json_file, "r", encoding="utf-8") as f:
            data = cls._data_model.model_validate_json(f.read())
        return cls(data)

    # 3. 从json文件种读取多个材料数据
    @classmethod
    def from_jsons(cls, json_file: str) -> list["Material"]:
        """从 JSON 创建实例"""
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.loads(f.read())

        return [cls(cls._data_model.model_validate(item)) for item in data]

    # 3. 从dict字典读取材料数据
    @classmethod
    def from_dict(cls, dict_data: dict) -> "Material":
        """从 dict 创建实例"""
        data = cls._data_model.model_validate(dict_data)
        return cls(data)

    # 4. 从dict字典读取多个材料数据
    @classmethod
    def from_dicts(cls, dict_data: list[dict]) -> list["Material"]:
        """从 dict 创建实例"""
        return [cls(cls._data_model.model_validate(item)) for item in dict_data]

    # 5. 将所有实例序列化为 JSON
    @classmethod
    def to_jsons(cls, json_file: str):
        """将所有实例序列化为 JSON"""
        data = [instance.to_dict() for instance in cls._instances]
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=4, exclude={"fraction"}))

    # 6. 获取所有的 Material 实例
    @classmethod
    def all(cls):
        return cls._instances

    # 7. 查找 Material 实例
    @classmethod
    def find(cls, name=None, abbreviation=None, cas_registry_number=None):
        if not any([name, abbreviation, cas_registry_number]):
            raise ValueError("At least one search parameter must be provided.")

        for instance in cls._instances:
            if (
                (name and instance.name == name)
                or (abbreviation and instance.abbreviation == abbreviation)
                or (
                    cas_registry_number
                    and instance.cas_registry_number == cas_registry_number
                )
            ):
                return instance

        raise ValueError("Material not found.")

    # ------------------------ 实例方法 ------------------------ #
    # 1. 序列化为 JSON
    def to_json(self, json_file: str):
        """序列化为 JSON"""
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(self._data.model_dump_json(indent=4, exclude={"fraction"}))

    # 3. 序列化为 dict
    def to_dict(self) -> dict:
        """序列化为 dict"""
        return self._data.model_dump(exclude={"fraction"})

    # 4. 将所有实例序列化为 dict
    @classmethod
    def to_dicts(cls) -> list[dict]:
        """将所有实例序列化为 dict"""
        return [instance.to_dict() for instance in cls._instances]


# 子类：锂盐（LithiumSalt）
class LithiumSalt(Material[LithiumSaltModel]):
    # 类变量，用于保存所有实例
    _instances: ClassVar[list["LithiumSalt"]] = []
    _data_model = LithiumSaltModel

    def __init__(self, data: LithiumSaltModel):
        super().__init__(data)

    # ------------------------ 魔术方法 ------------------------ #
    # ------------------------ 私有方法 ------------------------ #
    # ------------------------ 属性方法 ------------------------ #
    # ------------------------ 类方法 ------------------------ #
    # 1. 使用数据模型创建锂盐实例
    @classmethod
    def create(
        cls,
        name,
        abbreviation,
        cas_registry_number,
        description,
        molecular_structure,
        density,
        melting_point,
        boiling_point,
        solubility,
        anion_size,
        dissociation_constant,
        thermal_stability,
        electrochemical_stability,
    ) -> "Material":
        """使用数据模型创建锂盐实例"""

        data = LithiumSaltModel(
            name=name,
            abbreviation=abbreviation,
            material_type=MaterialType.LITHIUM_SALT,
            cas_registry_number=cas_registry_number,
            description=description,
            molecular_structure=molecular_structure,
            density=density,
            melting_point=melting_point,
            boiling_point=boiling_point,
            solubility=solubility,
            anion_size=anion_size,
            dissociation_constant=dissociation_constant,
            thermal_stability=thermal_stability,
            electrochemical_stability=electrochemical_stability,
        )
        return cls(data)

    # ------------------------ 实例方法 ------------------------ #


# 子类：溶剂（Solvent）
class Solvent(Material[SolventModel]):
    # 类变量，用于保存所有实例
    _instances: ClassVar[list["Solvent"]] = []
    _data_model = SolventModel

    def __init__(
        self,
        data: SolventModel,
    ):
        # 调用基类的构造函数
        super().__init__(data)

    # ------------------------ 魔术方法 ------------------------ #
    # ------------------------ 私有方法 ------------------------ #
    # ------------------------ 属性方法 ------------------------ #
    # ------------------------ 类方法 ------------------------ #
    # 1. 使用数据模型创建溶剂实例
    @classmethod
    def create(
        cls,
        name,
        abbreviation,
        cas_registry_number,
        description,
        molecular_structure,
        density,
        melting_point,
        boiling_point,
        dielectric_constant,
        viscosity,
        dipole_moment,
        electrochemical_window,
        hydrogen_bonding,
    ) -> "Solvent":
        """使用数据模型创建溶剂实例"""

        data = SolventModel(
            name=name,
            abbreviation=abbreviation,
            material_type=MaterialType.SOLVENT,
            cas_registry_number=cas_registry_number,
            description=description,
            molecular_structure=molecular_structure,
            density=density,
            melting_point=melting_point,
            boiling_point=boiling_point,
            dielectric_constant=dielectric_constant,
            viscosity=viscosity,
            dipole_moment=dipole_moment,
            electrochemical_window=electrochemical_window,
            hydrogen_bonding=hydrogen_bonding,
        )
        return cls(data)

    # ------------------------ 实例方法 ------------------------ #


# 子类：添加剂（Additive）
class Additive(Material):
    # 类变量，用于保存所有实例
    _instances: ClassVar[list["Additive"]] = []
    _data_model = AdditiveModel

    def __init__(self, data: AdditiveModel):
        super().__init__(data)

    # ------------------------ 魔术方法 ------------------------ #
    # ------------------------ 私有方法 ------------------------ #
    # ------------------------ 属性方法 ------------------------ #
    # ------------------------ 类方法 ------------------------ #
    # 1. 使用数据模型创建添加剂实例
    @classmethod
    def create(
        cls,
        name,
        abbreviation,
        cas_registry_number,
        description,
        molecular_structure,
        density,
        melting_point,
        boiling_point,
        reduction_potential,
        oxidation_potential,
        concentration,
        action,
    ) -> "Additive":
        """使用数据模型创建添加剂实例"""

        data = AdditiveModel(
            name=name,
            abbreviation=abbreviation,
            material_type=MaterialType.ADDITIVE,
            cas_registry_number=cas_registry_number,
            description=description,
            molecular_structure=molecular_structure,
            density=density,
            melting_point=melting_point,
            boiling_point=boiling_point,
            reduction_potential=reduction_potential,
            oxidation_potential=oxidation_potential,
            concentration=concentration,
            action=action,
        )
        return cls(data)

    # ------------------------ 实例方法 ------------------------ #
