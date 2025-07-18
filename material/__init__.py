from functools import cached_property
import json
from typing import Dict, Optional, Union
from rdkit.Chem import MACCSkeys, Descriptors, MolFromSmiles, rdMolDescriptors

from config import MLIBRARY_PATH, MLIBRARY_STRICT_MODE

from material.models import (
    MaterialModel,
    SaltModel,
    SolventModel,
    AdditiveModel,
)


# 材料（Material）
class Material:
    __slots__: tuple[str, ...] = (
        "_data",
        "_mol",
        "_fingerprint",
        "_molecular_weight",
        "ml_features",
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

    # 1. 计算分子指纹
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
        fingerprint = MACCSkeys._pyGenMACCSKeys(self._mol)

        # 将指纹转换为NumPy数组并转换为列表，供机器学习使用
        return list(fingerprint)

    # 2. 计算分子量
    @cached_property
    def molecular_weight(self):
        """
        计算分子量。
        """
        # return Descriptors.CalcMolDescriptors(mol).get("MolWt")
        return Descriptors.MolWt(self._mol)  # type: ignore

    # 3. 获取材料分子信息
    @property
    def molecular_descriptor(self) -> dict:
        """
        返回材料的分子信息。
        """

        return Descriptors.CalcMolDescriptors(self._mol)

    # 4. 获取材料分子式
    @property
    def molecular_formula(self) -> str:
        """
        返回材料的分子式。
        """
        mol = MolFromSmiles(self.molecular_structure)
        return rdMolDescriptors.CalcMolFormula(mol)

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


class MaterialLibrary:
    # 类型映射字典
    _TYPE_MAPPINGS: dict[str, type[MaterialModel]] = {
        "Salt": SaltModel,
        "Solvent": SolventModel,
        "Additive": AdditiveModel,
    }

    def __init__(self, file_path: Optional[str] = None):
        """
        初始化材料库。

        Args:
            file_path (str, optional): JSON 文件路径，用于加载材料数据。
        """
        self.file_path = file_path
        self.pool: Dict[str, Material] = {}  # 已创建材料实例的池
        self.material_index: Dict[str, dict] = {}  # 材料数据索引
        # 定义需要标准化属性列表
        self.salt_attrs = {
            "density": [],
            "melting_point": [],
            "solubility": [],
            "anion_size": [],
            "dissociation_constant": [],
            "thermal_stability": [],
            "electrochemical_stability": [],
        }
        self.solvent_attrs = {
            "density": [],
            "melting_point": [],
            "boiling_point": [],
            "dielectric_constant": [],
            "viscosity": [],
            "dipole_moment": [],
            "electrochemical_window": [],
        }
        self.additive_attrs = {
            "density": [],
            "melting_point": [],
            "boiling_point": [],
            "reduction_potential": [],
            "oxidation_potential": [],
        }
        if file_path:
            self._build_index()
            self._precompute_standardized_values()

    def __len__(self) -> int:
        """返回材料库中的材料总数"""
        return len(self.material_index)

    def _build_index(self):
        """从 JSON 文件构建材料索引"""
        assert self.file_path is not None, "file_path must not be None"
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.material_index = json.load(f)

    def _precompute_standardized_values(self):
        """对所有材料属性进行标准化"""

        # 收集所有材料的属性值
        for cas_number, data in self.material_index.items():
            material_type = data["material_type"]
            if material_type == "Salt":
                for attr in self.salt_attrs:
                    value = data.get(attr)
                    if self.salt_attrs[attr] == []:
                        self.salt_attrs[attr].append(value if value is not None else 0)
                        self.salt_attrs[attr].append(value if value is not None else 0)
                    else:
                        self.salt_attrs[attr][0] = min(
                            self.salt_attrs[attr][0], value if value is not None else 0
                        )
                        self.salt_attrs[attr][1] = max(
                            self.salt_attrs[attr][1], value if value is not None else 0
                        )
            elif material_type == "Solvent":
                for attr in self.solvent_attrs:
                    value = data.get(attr)
                    if self.solvent_attrs[attr] == []:
                        self.solvent_attrs[attr].append(
                            value if value is not None else 0
                        )
                        self.solvent_attrs[attr].append(
                            value if value is not None else 0
                        )
                    else:
                        self.solvent_attrs[attr][0] = min(
                            self.solvent_attrs[attr][0],
                            value if value is not None else 0,
                        )
                        self.solvent_attrs[attr][1] = max(
                            self.solvent_attrs[attr][1],
                            value if value is not None else 0,
                        )
            elif material_type == "Additive":
                for attr in self.additive_attrs:
                    value = data.get(attr)
                    if self.additive_attrs[attr] == []:
                        self.additive_attrs[attr].append(
                            value if value is not None else 0
                        )
                        self.additive_attrs[attr].append(
                            value if value is not None else 0
                        )
                    else:
                        self.additive_attrs[attr][0] = min(
                            self.additive_attrs[attr][0],
                            value if value is not None else 0,
                        )
                        self.additive_attrs[attr][1] = max(
                            self.additive_attrs[attr][1],
                            value if value is not None else 0,
                        )

    def get_standardized_value(self, material: Material) -> dict:
        """
        获取标准化值。
        """
        epsilon = 1e-8  # 防止除零的小偏移量

        if material.material_type == "Salt":
            standardized_values = {
                attr: (getattr(material, attr) - self.salt_attrs[attr][0])
                / (self.salt_attrs[attr][1] - self.salt_attrs[attr][0] + epsilon)
                for attr in self.salt_attrs
            }
        elif material.material_type == "Solvent":
            standardized_values = {
                attr: (getattr(material, attr) - self.solvent_attrs[attr][0])
                / (self.solvent_attrs[attr][1] - self.solvent_attrs[attr][0] + epsilon)
                for attr in self.solvent_attrs
            }
        elif material.material_type == "Additive":
            standardized_values = {
                attr: (getattr(material, attr) - self.additive_attrs[attr][0])
                / (
                    self.additive_attrs[attr][1]
                    - self.additive_attrs[attr][0]
                    + epsilon
                )
                for attr in self.additive_attrs
            }
        else:
            raise ValueError(f"Unknown material type: {material.material_type}")

        return standardized_values

    def get_material(
        self, abbr: str, cas_registry_number: str, strict=MLIBRARY_STRICT_MODE
    ) -> Material:
        """
        按需加载材料。

        Args:
            abbr (str): 材料简称。
            cas_registry_number (str): 材料CAS号。

        Returns:
            Union[Salt, Solvent, Additive]: 对应的材料实例。

        Raises:
            ValueError: 如果材料不存在或类型未知。
        """

        if cas_registry_number not in self.material_index:
            raise ValueError(
                f"Material {abbr}[{cas_registry_number}] not found in library."
            )

        material: dict = self.material_index[cas_registry_number]
        if material["abbreviation"].lower() != abbr.lower() and strict:
            raise ValueError(
                f"Material [{cas_registry_number}] found in library but is named {material['abbr']}."
            )

        if cas_registry_number not in self.pool:
            material_type: str = material["material_type"]
            model = self._TYPE_MAPPINGS[material_type](**material)
            self.pool[cas_registry_number] = Material(model)

        return self.pool[cas_registry_number]

    def add_material(self, data: Union[dict, MaterialModel]) -> Material:
        """
        添加新材料到缓存和索引。

        Args:
            data (dict): 材料数据。

        Returns:
            Material: 新创建的材料实例。
        """

        material_type: str = (
            data["material_type"]
            if isinstance(data, dict)
            else data.material_type.value
        )
        cas_registry_number: str = (
            data["cas_registry_number"]
            if isinstance(data, dict)
            else data.cas_registry_number
        )
        model: MaterialModel = (
            self._TYPE_MAPPINGS[material_type][0](**data)
            if isinstance(data, dict)
            else data
        )

        self.material_index[cas_registry_number] = model.model_dump()
        self.pool[cas_registry_number] = Material(model)

        return self.pool[cas_registry_number]

    def save(self, file_path: str):
        """
        保存所有材料到 JSON 文件。

        Args:
            file_path (str): 保存路径。
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.material_index, f, indent=4)


MLibrary = MaterialLibrary(MLIBRARY_PATH)
