from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional
from enum import Enum


# 定义枚举类来表示材料类型
class MaterialType(Enum):
    SOLVENT = "Solvent"
    SALT = "Salt"
    ADDITIVE = "Additive"


# 氢键能力枚举
class HydrogenBonding(Enum):
    STRONG = 1
    WEAK = 0.5
    NONE = 0


# 添加剂作用枚举（可选）
class AdditiveAction(str, Enum):
    SEI_FORMATION = "SEI formation"
    FLAME_RETARDANT = "flame retardant"
    STABILIZER = "stabilizer"
    OTHER = "other"


# 基础材料数据模型
class MaterialModel(BaseModel):
    name: str = Field(..., min_length=1, description="材料名称")
    abbreviation: str = Field(..., min_length=1, description="材料缩写")
    material_type: MaterialType = Field(..., description="材料类型")
    cas_registry_number: str = Field(
        ..., pattern=r"^\d{2,7}-\d{2}-\d$", description="CAS注册号，格式如 123-45-6"
    )
    description: str = Field(..., min_length=1, description="材料描述")
    use_as_standard_data: bool = Field(
        default=False, description="是否作为标准化计算数据"
    )
    molecular_structure: str = Field(
        ..., min_length=1, description="分子结构 (SMILES格式)"
    )
    density: float = Field(..., gt=0, description="密度 (g/cm³)")
    melting_point: float = Field(..., description="熔点 (°C)")
    boiling_point: Optional[float] = Field(None, description="沸点 (°C，可选)")

    # 配置模型行为
    model_config = ConfigDict(
        use_enum_values=True,  # 使用枚举值而不是名称
    )

    # 验证器
    @field_validator("abbreviation")
    def abbreviation_lowercase(cls, value):
        """确保缩写为小写，与主类一致"""
        return value.lower()

    @field_validator("cas_registry_number")
    def validate_cas_format(cls, value):
        """验证CAS号格式"""
        if not value.replace("-", "").isdigit():
            raise ValueError("CAS registry number 只能包含数字和短横线")
        return value

    @field_validator("density")
    def validate_density(cls, value):
        """确保密度为正"""
        if value <= 0:
            raise ValueError("密度必须为正")
        return value


# 锂盐数据模型
class SaltModel(MaterialModel):
    solubility: float = Field(..., gt=0, description="溶解度 (mol/L)")
    anion_size: float = Field(..., gt=0, description="阴离子尺寸 (Å)")
    dissociation_constant: float = Field(..., ge=0, description="解离常数 (无单位)")
    thermal_stability: float = Field(..., description="热稳定性，分解温度 (°C)")
    electrochemical_stability: float = Field(
        ..., gt=0, description="电化学稳定性，电压窗口 (V)"
    )
    material_type: MaterialType = MaterialType.SALT
    # 配置模型行为
    model_config = ConfigDict(
        use_enum_values=True,  # 使用枚举值而不是名称
    )

    # 验证器
    @field_validator("solubility")
    def validate_solubility(cls, value):
        if value > 100:  # 假设溶解度上限为 100 mol/L
            raise ValueError(f"Solubility too high: {value} mol/L")
        return value

    @field_validator("anion_size")
    def validate_anion_size(cls, value):
        if not 0.1 <= value <= 10:  # 假设阴离子尺寸范围为 0.1-10 Å
            raise ValueError(f"Anion size must be between 0.1 and 10 Å, got {value}")
        return value

    @field_validator("dissociation_constant")
    def validate_electrochemical_stability(cls, value):
        if not 0 < value <= 10:  # 假设电压窗口范围为 0-10 V
            raise ValueError(
                f"Electrochemical stability must be between 0 and 10 V, got {value}"
            )
        return value


# 溶剂数据模型
class SolventModel(MaterialModel):
    dielectric_constant: float = Field(..., gt=0, description="介电常数 (无单位)")
    viscosity: float = Field(..., gt=0, description="粘度 (mPa·s)")
    dipole_moment: float = Field(..., ge=0, description="偶极矩 (Debye)")
    electrochemical_window: float = Field(..., gt=0, description="电化学窗口 (V)")
    hydrogen_bonding: HydrogenBonding = Field(
        ..., description="氢键能力 (strong/weak/none)"
    )
    material_type: MaterialType = MaterialType.SOLVENT
    # 配置模型行为
    model_config = ConfigDict(
        use_enum_values=True,  # 使用枚举值而不是名称
    )

    # 验证器
    @field_validator("dielectric_constant")
    def validate_dielectric_constant(cls, value):
        if not 1 <= value <= 1000:  # 假设介电常数范围为 1-1000
            raise ValueError(
                f"Dielectric constant must be between 1 and 1000, got {value}"
            )
        return value

    @field_validator("viscosity")
    def validate_viscosity(cls, value):
        if not 0.1 <= value <= 100:  # 假设粘度范围为 0.1-100 mPa·s
            raise ValueError(
                f"Viscosity must be between 0.1 and 100 mPa·s, got {value}"
            )
        return value

    @field_validator("electrochemical_window")
    def validate_electrochemical_window(cls, value):
        if not 0 < value <= 10:  # 假设电化学窗口范围为 0-10 V
            raise ValueError(
                f"Electrochemical window must be between 0 and 10 V, got {value}"
            )
        return value


# 添加剂数据模型
class AdditiveModel(MaterialModel):
    reduction_potential: float = Field(..., description="还原电位 (V)")
    oxidation_potential: float = Field(..., description="氧化电位 (V)")
    action: str = Field(..., description="添加剂作用（如 'SEI formation'）")
    material_type: MaterialType = MaterialType.ADDITIVE
    # 配置模型行为
    model_config = ConfigDict(
        use_enum_values=True,  # 使用枚举值而不是名称
    )

    # 验证器
    @field_validator("reduction_potential")
    def validate_reduction_potential(cls, value):
        if not -5 <= value <= 5:  # 假设还原电位范围为 -5 到 5 V
            raise ValueError(
                f"Reduction potential must be between -5 and 5 V, got {value}"
            )
        return value

    @field_validator("oxidation_potential")
    def validate_oxidation_potential(cls, value):
        if not 0 <= value <= 10:  # 假设氧化电位范围为 0-10 V
            raise ValueError(
                f"Oxidation potential must be between 0 and 10 V, got {value}"
            )
        return value

    @field_validator("action")
    def validate_action(cls, value):
        # 可选：严格限制为枚举值
        allowed_actions = {"SEI formation", "flame retardant", "stabilizer", "other"}
        if value not in allowed_actions:
            raise ValueError(f"Action must be one of {allowed_actions}, got {value}")
