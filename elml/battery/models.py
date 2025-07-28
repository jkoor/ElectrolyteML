from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional

from ..material.models import (
    MaterialModel,
)


# 成分类：表示材料及其占比
class Component(BaseModel):
    abbr: str
    cas_registry_number: str
    overall_fraction: Optional[float] = Field(
        default=None, ge=0, le=100, description="整体占比"
    )
    relative_fraction: Optional[float] = Field(
        default=None, ge=0, le=100, description="相对占比"
    )


# 电解液数据模型
class ElectrolyteModel(BaseModel):
    name: str
    id: str = Field(default="", description="电解液配方ID")
    description: str = Field(..., min_length=1, description="电解液配方名称")
    salts: list[Component] = Field(..., description="锂盐列表")
    solvents: list[Component] = Field(..., description="溶剂列表")
    additives: list[Component] = Field(default_factory=list, description="添加剂列表")
    # 配置模型行为
    model_config = ConfigDict(
        use_enum_values=True,  # 使用枚举值而不是名称
    )
    condition: dict[str, float] = Field(..., description="电解液配方条件")
    performance: dict[str, float] = Field(
        ..., description="电解液性能参数，如电导率、离子传输数等"
    )

    @model_validator(mode="after")
    def compute_fraction(self):
        """计算电解液配方的总百分比"""

        # 计算锂盐总占比
        salt_fraction = sum(
            component.overall_fraction
            for component in self.salts
            if component.overall_fraction is not None
        )
        # 计算添加剂总占比
        additive_fraction = sum(
            component.overall_fraction
            for component in self.additives
            if component.overall_fraction is not None
        )

        # 计算溶剂总占比
        solvent_fraction = 100 - salt_fraction - additive_fraction

        # 计算锂盐相对占比
        for component in self.salts:
            if component.overall_fraction is not None:
                component.relative_fraction = round(
                    component.overall_fraction / salt_fraction * 100, 2
                )

        # 计算添加剂相对占比
        for component in self.additives:
            if component.overall_fraction is not None:
                component.relative_fraction = round(
                    component.overall_fraction / additive_fraction * 100, 2
                )

        # 验证溶剂各组分相对占比是否为 100%
        solvent_relative_fraction = sum(
            component.relative_fraction
            for component in self.solvents
            if component.relative_fraction is not None
        )
        if 100 - solvent_relative_fraction >= 0.1:
            raise ValueError("溶剂组分相对占比必须为 100%")

        # 计算溶剂组分绝对占比
        for component in self.solvents:
            if component.relative_fraction is not None:
                component.overall_fraction = round(
                    component.relative_fraction * solvent_fraction * 0.01, 2
                )
            else:
                raise ValueError("必须提供溶剂的相对占比")

        return self

    # 验证condition是否包含温度
    @model_validator(mode="after")
    def validate_condition(self):
        if "temperature" not in self.condition:
            self.condition["temperature"] = 298.15
        return self


# 正极数据模型
class CathodeModel(BaseModel):
    name: str
    id: str
    description: str = Field(..., min_length=1, description="正极材料描述")
    material: MaterialModel = Field(..., description="正极材料")
    performance: dict = Field(..., description="正极材料性能参数")


# 负极数据模型
class AnodeModel(BaseModel):
    name: str
    id: str
    description: str = Field(..., min_length=1, description="负极材料描述")
    material: MaterialModel = Field(..., description="负极材料")
    performance: dict = Field(..., description="负极材料性能参数")


# 电池数据模型
class BatteryModel(BaseModel):
    name: str
    id: str = Field(default="", description="电池ID，由电解液、正极和负极ID组合而成")

    description: str = Field(..., min_length=1, description="电池描述")
    electrolyte: ElectrolyteModel = Field(..., description="电解液配方")
    cathode: CathodeModel = Field(..., description="正极材料")
    anode: AnodeModel = Field(..., description="负极材料")
    performance: dict = Field(..., description="电池性能参数")

    @model_validator(mode="after")
    def compute_id(self):
        self.id = f"{self.electrolyte.id}-{self.cathode.id}-{self.anode.id}"
        return self
