from typing import Optional

import pandas as pd

from elml import MLibrary
from elml.battery import Electrolyte
from elml.material import Material
from elml.dataset import ElectrolyteDataset


def solvent_weight_ratio(
    solvents: list[tuple[Material, float]],
    unit_type: str = "w",  # w: weight, v: volume, mol: molar
) -> list[tuple[Material, float]]:
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
        components: list[tuple[Material, float]] = [
            (solvent, round(ratio / total_mass * 100, 2)) for solvent, ratio in solvents
        ]
    # 输入为体积比，总和为 1
    elif unit_type == "v":
        total_mass = sum(ratio * solvent.density for solvent, ratio in solvents)
        components: list[tuple[Material, float]] = [
            (solvent, round(ratio * solvent.density / total_mass * 100, 2))
            for solvent, ratio in solvents
        ]
    elif unit_type == "mol":
        total_mass = sum(
            ratio * solvent.molecular_weight for solvent, ratio in solvents
        )
        components: list[tuple[Material, float]] = [
            (
                solvent,
                round(ratio * solvent.molecular_weight / total_mass * 100, 2),
            )
            for solvent, ratio in solvents
        ]
    else:
        raise ValueError(f"unit_type 参数错误: {unit_type}")

    return components


def salt_weight_ratio(
    salts: list[tuple[Material, float]],
    unit_type: str = "w",  # w: mol/kg, v: mol/L
    solvents: Optional[list[tuple[Material, float]]] = None,
) -> list[tuple[Material, float]]:
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
        components: list[tuple[Material, float]] = [
            (
                salt,
                round((ratio * salt.molecular_weight / 10), 2),
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
                round((ratio * salt.molecular_weight / (total_mass * 10)), 2),
            )
            for salt, ratio in salts
        ]

    return components


filepath = "data/calisol23/calisol23.csv"

# 读取数据
df = pd.read_csv(filepath)

# 筛选特定字段值
filter_salt = ["LIPF6", "LIFSI", "LITFSI", "LIBF4"]
# filter_salt = ["LIBOB"]
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

dataset = ElectrolyteDataset()

# 创建与df完全相同的空表
valid_df = df.iloc[0:0].copy()  # 保留列结构但不包含数据行

# 按行遍历数据
for idx, row in df.iterrows():
    if pd.isna(row["k"]) or row["k"] <= 0:
        continue

    # 获取锂盐及其比例
    salt: Material = MLibrary.get_material(abbr=str(row["salt"]))

    # 获取所有非零溶剂及其比例
    has_unknown_solvent = False  # 是否有未知溶剂
    solvents_init: list[tuple[Material, float]] = []
    solvent_columns = df.columns[8:48]  # 从EC列到o_Xylene列
    for abbr in solvent_columns:
        # 如果该溶剂的比例不为0，则添加到列表中
        if row[abbr] != 0:
            # 如果数据库中没有该溶剂的简称，则跳过该配方
            if not MLibrary.has_material(abbr=abbr):
                has_unknown_solvent = True
                continue
            solvent: Material = MLibrary.get_material(abbr=abbr)
            solvents_init.append((solvent, row[abbr]))

    if has_unknown_solvent or solvents_init == []:
        continue

    # 将符合条件的数据行添加到空表中
    valid_df = pd.concat([valid_df, pd.DataFrame([row])], ignore_index=True)

    # 计算溶剂重量比例
    solvents: list[tuple[Material, float]] = solvent_weight_ratio(
        solvents_init, row["solvent_ratio_type"]
    )

    # 计算锂盐重量比例
    salts_init: list[tuple[Material, float]] = [(salt, row["c"])]

    if salts_init[0][1] <= 0:
        continue

    if row["c_unit"] == "mol/kg":
        salt_unit_type = "w"
    elif row["c_unit"] == "mol/l":
        salt_unit_type = "v"
    else:
        raise ValueError(f"unit_type 参数错误: {row['c_unit']}")

    salts: list[tuple[Material, float]] = salt_weight_ratio(
        salts_init, salt_unit_type, solvents
    )

    el: Electrolyte = Electrolyte.create(
        name=f"{salt.abbreviation} + {', '.join(solvent.abbreviation for solvent, _ in solvents)}",
        id=str(row["c"]),
        description=row["doi"],
        salts=salts,
        solvents=solvents,
        additives=[],
        performance={
            "conductivity": row["k"],
        },
        condition={"temperature": row["T"]},
    )

    dataset.add_formula(el)


json_filepath = filepath.replace(".csv", ".json")
dataset.to_json(json_filepath)

# 保存包含有效电解质配方的DataFrame
valid_filepath = filepath.replace(".csv", "_valid.csv")
valid_df.to_csv(valid_filepath, index=False)
print(f"原始数据行数: {len(df)}")
print(f"有效配方行数: {len(valid_df)}")
print(f"有效配方保存至: {valid_filepath}")
