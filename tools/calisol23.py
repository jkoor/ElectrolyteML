from typing import Optional

import pandas as pd

from source.SALT import LIPF6, LIFSI, LITFSI, LIBF4
from source.SOLVENT import EC, DMC, DEC, PC, EMC, DME, EA, MA
from src.models.battery import Electrolyte
from src.models.materials import Solvent, Salt
from src.models.base import SolventModel, SaltModel


def solvent_weight_ratio(
    solvents: list[tuple[SolventModel, float]],
    unit_type: str = "w",  # w: weight, v: volume, mol: molar
) -> list[tuple[SolventModel, float]]:
    """
    将溶剂的体积比转换为质量比

    参数:
    solvents: 包含元组的列表，每个元组包含溶剂对象和体积比
    unit_type: 转换类型，'w' 代表质量，'v' 代表体积

    返回:
    components: 包含组件对象的列表
    """
    # 输入为质量比，总和为 1
    if unit_type == "w":
        total_mass = sum(ratio for solvent, ratio in solvents)
        components: list[tuple[SolventModel, float]] = [
            (solvent, round(ratio / total_mass * 100, 2)) for solvent, ratio in solvents
        ]
    # 输入为体积比，总和为 1
    elif unit_type == "v":
        total_mass = sum(ratio * solvent.density for solvent, ratio in solvents)
        components: list[tuple[SolventModel, float]] = [
            (solvent, round(ratio * solvent.density / total_mass * 100, 2))
            for solvent, ratio in solvents
        ]
    elif unit_type == "mol":
        total_mass = sum(
            ratio * Solvent(solvent).molecular_weight for solvent, ratio in solvents
        )
        components: list[tuple[SolventModel, float]] = [
            (
                solvent,
                round(ratio * Solvent(solvent).molecular_weight / total_mass * 100, 2),
            )
            for solvent, ratio in solvents
        ]
    else:
        raise ValueError(f"unit_type 参数错误: {unit_type}")

    return components


def salt_weight_ratio(
    salts: list[tuple[SaltModel, float]],
    unit_type: str = "w",  # w: mol/kg, v: mol/L
    solvents: Optional[list[tuple[SolventModel, float]]] = None,
) -> list[tuple[SaltModel, float]]:
    """
    将摩尔占比转换为重量占比

    参数:
    mole_percentages: 锂盐的摩尔比(mol/kg) (mol/L)
    molecular_weights: 锂盐的分子量(g/mol)

    返回:
    weight_percentages: 物质的重量百分比(%)
    """

    # 输入为摩尔质量比
    if unit_type == "w":
        components: list[tuple[SaltModel, float]] = [
            (
                salt,
                round((ratio * Salt(salt).molecular_weight / 10), 2),
            )
            for salt, ratio in salts
        ]
    # 输入为摩尔体积比
    elif unit_type == "v":
        total_mass = 0  # kg
        if solvents:
            for solvent, fraction in solvents:
                total_mass += solvent.density * fraction * 0.01
        components = [
            (
                salt,
                round((ratio * Salt(salt).molecular_weight / (total_mass * 10)), 2),
            )
            for salt, ratio in salts
        ]

    return components


MaterialMAP = {
    "LIPF6": LIPF6,
    "LIFSI": LIFSI,
    "LITFSI": LITFSI,
    "LIBF4": LIBF4,
    "EC": EC,
    "DMC": DMC,
    "DEC": DEC,
    "PC": PC,
    "EMC": EMC,
    "DME": DME,
    "EA": EA,
    "MA": MA,
}

# 读取数据
df = pd.read_csv("database/calisol23_DOI_10.11583DTU.c.6929599.csv")

# 筛选特定字段值
filter_salt = ["LIPF6", "LIFSI", "LITFSI", "LIBF4"]
df = df[df["salt"].str.upper().isin(filter_salt)]

# 重命名列名
df.columns = [
    "i",
    "doi",
    "k",
    "T",
    "c",
    "salt",
    "c_unit",
    "solvent_ratio_type",
    "EC",
    "PC",
    "DMC",
    "EMC",
    "DEC",
    "DME",
    "DMSO",
    "AN",
    "MOEMC",
    "TFP",
    "EA",
    "MA",
    "FEC",
    "DOL",
    "MeTHF_2",
    "DMM",
    "Freon_11",
    "Methylene_chloride",
    "THF",
    "Toluene",
    "Sulfolane",
    "Glyme_2",
    "Glyme_3",
    "Glyme_4",
    "Me_3_Oxazolidinone_2",
    "MeSulfolane_3",
    "Ethyldiglyme",
    "DMF",
    "Ethylbenzene",
    "Ethylmonoglyme",
    "Benzene",
    "g_Butyrolactone",
    "Cumene",
    "Propylsulfone",
    "Pseudocumeme",
    "TEOS",
    "m_Xylene",
    "o_Xylene",
]

# 按行遍历数据
for row in df.itertuples():
    # 获取锂盐及其比例
    salt_model: SaltModel = MaterialMAP[str(getattr(row, "salt")).upper()]

    # 获取所有非零溶剂及其比例
    solvents_init: list[tuple[SolventModel, float]] = []
    solvent_columns = df.columns[8:48]  # 从EC列到o_Xylene列
    for solvent in solvent_columns:
        if getattr(row, solvent) != 0 and solvent in MaterialMAP:
            solvent_model: SolventModel = MaterialMAP[solvent]
            solvents_init.append((solvent_model, getattr(row, solvent)))

    if not solvents_init:
        continue

    # 计算溶剂重量比例
    solvents: list[tuple[SolventModel, float]] = solvent_weight_ratio(
        solvents_init, getattr(row, "solvent_ratio_type")
    )

    # 计算锂盐重量比例
    salts_init: list[tuple[SaltModel, float]] = [(salt_model, getattr(row, "c"))]

    if salts_init[0][1] <= 0:
        continue

    if getattr(row, "c_unit") == "mol/kg":
        salt_unit_type = "w"
    elif getattr(row, "c_unit") == "mol/l":
        salt_unit_type = "v"
    else:
        raise ValueError(f"unit_type 参数错误: {getattr(row, 'c_unit')}")

    salts: list[tuple[SaltModel, float]] = salt_weight_ratio(
        salts_init, salt_unit_type, solvents
    )

    Electrolyte.create(
        name=f"{salt_model.abbreviation} + {', '.join(solvent.abbreviation for solvent, _ in solvents)}",
        id=str(getattr(row, "c")),
        description=getattr(row, "doi"),
        salts=salts,
        solvents=solvents,
        additives=[],
        performance={
            "conductivity": getattr(row, "k"),
            "temperature": getattr(row, "T"),
        },
    )

# Electrolyte.to_jsons("database/electrolyte.json")

# ec = (EC, 1)
# emc = (EMC, 1)
# result_0 = solvent_weight_ratio([ec, emc], "v")
# result_1 = salt_weight_ratio([(LIPF6, 1)], "v", result_0)
pass
