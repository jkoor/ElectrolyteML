import json
from typing import Dict, Optional, Union

from ..material import Material
from .models import (
    MaterialModel,
    SaltModel,
    SolventModel,
    AdditiveModel,
)


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
        self.abbr_to_cas_index: Dict[str, str] = {}  # 简称到CAS号的映射
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
            self.abbr_to_cas_index = {
                data["abbreviation"].lower(): cas_number
                for cas_number, data in self.material_index.items()
            }

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

    def has_material(
        self, abbr: Union[str, None] = None, cas_registry_number: Union[str, None] = None
    ) -> bool:
        """
        检查材料是否存在于库中。
        Args:
            abbr (str): 材料简称。
            cas_registry_number (str): 材料CAS号。
            至少提供一个参数。
        Returns:
            bool: 如果材料存在，返回 True；否则返回 False。
        """
        if not abbr and not cas_registry_number:  # abbr x, cas x
            raise ValueError("至少提供一个参数: abbr 或 cas_registry_number")
        elif not cas_registry_number and abbr:  # abbr o, cas x
            return abbr.lower() in self.abbr_to_cas_index
        elif not abbr and cas_registry_number:  # abbr x, cas o
            return cas_registry_number in self.material_index
        elif abbr and cas_registry_number:  # abbr o, cas o
            return (
                self.abbr_to_cas_index[abbr.lower()] == cas_registry_number
                if abbr.lower() in self.abbr_to_cas_index
                else False
            )

        return False

    def get_material(
        self,
        abbr: Union[str, None] = None,
        cas_registry_number: Union[str, None] = None,
    ) -> Material:
        """
        按需加载材料。

        Args:
            abbr (str): 材料简称。
            cas_registry_number (str): 材料CAS号。
            至少提供一个参数。

        Returns:
            Union[Salt, Solvent, Additive]: 对应的材料实例。

        Raises:
            ValueError: 如果材料不存在或类型未知。
        """

        # 检查参数，必须至少提供一个参数
        if not abbr and not cas_registry_number:  # abbr x, cas x
            raise ValueError("至少提供一个参数: abbr 或 cas_registry_number")
        # 如果只提供了简称，则尝试从简称映射获取CAS号
        if not cas_registry_number and abbr:  # abbr o, cas x
            cas_from_abbr = self.abbr_to_cas_index.get(abbr.lower())
            if not cas_from_abbr:
                raise ValueError(f"Material '{abbr}' not found in library.")
            cas_registry_number = cas_from_abbr

        if cas_registry_number not in self.material_index:
            raise ValueError(
                f"Material {abbr}[{cas_registry_number}] not found in library."
            )

        material: dict = self.material_index[cas_registry_number]
        # 如果提供了简称，检查简称是否匹配
        if abbr and material["abbreviation"].lower() != abbr.lower():
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
