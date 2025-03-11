from material import Solvent, LithiumSalt, Electrolyte

# 测试：创建不同类型的材料实例
ec = Solvent.create(
    name="碳酸乙烯酯",
    abbreviation="ec",
    cas_registry_number="96-49-1",
    description="A colorless liquid with a faint odor.",
    molecular_structure="C1COC(=O)O1",
    density=1.3214,  # g/cm³, 20°C, PubChem
    melting_point=36.4,  # °C, PubChem
    boiling_point=248,  # °C, PubChem
    dielectric_constant=89.6,  # 40°C, Landolt-Börnstein
    dipole_moment=4.9,  # Debye, 实验值
    viscosity=1.9,  # mPa·s, 40°C (熔融态), 文献近似
    electrochemical_window=4.5,  # V vs. Li/Li⁺, 文献估计
    hydrogen_bonding="weak",  # 羰基有一定氢键能力，但较弱
)

pc = Solvent.create(
    name="碳酸丙烯酯",
    abbreviation="PC",
    cas_registry_number="108-32-7",
    description="A colorless, odorless liquid used as a high-dielectric solvent in lithium-ion battery electrolytes.",
    molecular_structure="CC1COC(=O)O1",  # SMILES
    density=1.205,  # g/cm³, 20°C, 来源: PubChem
    melting_point=-48.8,  # °C, 来源: PubChem
    boiling_point=242,  # °C, 来源: PubChem
    dielectric_constant=64.92,  # 25°C, 来源: CRC Handbook of Chemistry and Physics
    dipole_moment=4.81,  # Debye, 来源: 实验值, 文献 (J. Phys. Chem.)
    viscosity=2.53,  # mPa·s, 25°C, 来源: 文献 (Electrolyte Data Collection)
    electrochemical_window=4.5,  # V vs. Li/Li⁺, 来源: 电池文献近似值
    hydrogen_bonding="weak",  # 羰基存在弱氢键能力，来源: 化学性质分析
)

dmc = Solvent.create(
    name="碳酸二甲酯",
    abbreviation="DMC",
    cas_registry_number="616-38-6",
    description="A colorless, flammable liquid with a mild ester-like odor, used as a low-viscosity solvent in electrolytes.",
    molecular_structure="COC(=O)OC",  # SMILES
    density=1.069,  # g/cm³, 20°C, 来源: PubChem
    melting_point=4.6,  # °C, 来源: PubChem
    boiling_point=90,  # °C, 来源: PubChem
    dielectric_constant=3.107,  # 25°C, 来源: CRC Handbook
    dipole_moment=0.91,  # Debye, 来源: 计算值 (DFT, B3LYP/6-31G)
    viscosity=0.59,  # mPa·s, 25°C, 来源: 文献 (Electrolyte Data Collection)
    electrochemical_window=4.5,  # V vs. Li/Li⁺, 来源: 电池文献近似值
    hydrogen_bonding="none",  # 无强氢键给体或受体，来源: 分子结构分析
)

emc = Solvent.create(
    name="碳酸甲乙酯",
    abbreviation="EMC",
    cas_registry_number="623-53-0",
    description="A colorless liquid with a low viscosity, commonly used to improve electrolyte conductivity.",
    molecular_structure="CCOC(=O)OC",  # SMILES
    density=1.006,  # g/cm³, 20°C, 来源: PubChem
    melting_point=-55,  # °C, 来源: PubChem
    boiling_point=107,  # °C, 来源: PubChem
    dielectric_constant=2.9,  # 25°C, 来源: 文献 (Electrolyte Data Collection)
    dipole_moment=0.95,  # Debye, 来源: 计算值 (DFT, B3LYP/6-31G)
    viscosity=0.65,  # mPa·s, 25°C, 来源: 文献 (Electrolyte Data Collection)
    electrochemical_window=4.5,  # V vs. Li/Li⁺, 来源: 电池文献近似值
    hydrogen_bonding="none",  # 无强氢键能力，来源: 分子结构分析
)

dec = Solvent.create(
    name="碳酸二乙酯",
    abbreviation="DEC",
    cas_registry_number="105-58-8",
    description="A colorless liquid with a mild odor, used as a low-viscosity solvent in battery electrolytes.",
    molecular_structure="CCOC(=O)OCC",  # SMILES
    density=0.975,  # g/cm³, 20°C, 来源: PubChem
    melting_point=-74,  # °C, 来源: PubChem
    boiling_point=126,  # °C, 来源: PubChem
    dielectric_constant=2.82,  # 25°C, 来源: CRC Handbook
    dipole_moment=0.97,  # Debye, 来源: 计算值 (DFT, B3LYP/6-31G)
    viscosity=0.75,  # mPa·s, 25°C, 来源: 文献 (Electrolyte Data Collection)
    electrochemical_window=4.5,  # V vs. Li/Li⁺, 来源: 电池文献近似值
    hydrogen_bonding="none",  # 无强氢键能力，来源: 分子结构分析
)

lipf6 = LithiumSalt.create(
    name="六氟磷酸锂",
    abbreviation="LiPF6",
    cas_registry_number="21324-40-3",
    description="A white crystalline powder widely used as a conductive salt in lithium-ion battery electrolytes.",
    molecular_structure="[Li+].[F-][P+](F)(F)(F)(F)F",  # SMILES简化表示
    density=1.5,  # g/cm³, 固态近似值, 来源: 文献估计
    melting_point=200,  # °C, 分解前熔点, 来源: CRC Handbook
    boiling_point=None,  # 无明确沸点，热分解, 来源: 化学性质
    solubility=1.0,  # mol/L, 在EC/DMC (1:1)中, 来源: 电池文献
    anion_size=2.6,  # Å, PF₆⁻半径, 来源: DFT计算近似 (J. Phys. Chem.)
    dissociation_constant=0.9,  # 无单位, 高解离近似值, 来源: 文献估计
    thermal_stability=80,  # °C, 分解温度, 来源: 实验值 (J. Power Sources)
    electrochemical_stability=4.5,  # V vs. Li/Li⁺, 来源: 电池文献
)

lifsi = LithiumSalt.create(
    name="二氟磺酰亚胺锂",
    abbreviation="LiFSI",
    cas_registry_number="171611-11-3",
    description="A white solid with high solubility and stability, used in advanced lithium battery electrolytes.",
    molecular_structure="[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F",  # SMILES表示
    density=1.8,  # g/cm³, 固态估计值, 来源: 文献近似
    melting_point=145,  # °C, 来源: 文献 (J. Electrochem. Soc.)
    boiling_point=None,  # 无明确沸点，分解, 来源: 化学性质
    solubility=2.0,  # mol/L, 在EC/DMC中, 来源: 电池文献
    anion_size=2.9,  # Å, FSI⁻半径, 来源: DFT计算 (Chem. Mater.)
    dissociation_constant=0.95,  # 无单位, 高解离近似值, 来源: 文献估计
    thermal_stability=200,  # °C, 分解温度, 来源: 实验值 (Electrochim. Acta)
    electrochemical_stability=5.0,  # V vs. Li/Li⁺, 来源: 电池文献
)

litfsi = LithiumSalt.create(
    name="双三氟甲基磺酰亚胺锂",
    abbreviation="LiTFSI",
    cas_registry_number="90076-65-6",
    description="A white crystalline solid with excellent thermal and electrochemical stability, used in high-performance lithium battery electrolytes.",
    molecular_structure="[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",  # SMILES表示
    density=1.33,  # g/cm³, 固态估计值, 来源: 文献近似 (J. Electrochem. Soc.)
    melting_point=234,  # °C, 来源: CRC Handbook
    boiling_point=None,  # 无明确沸点，热分解, 来源: 化学性质
    solubility=1.5,  # mol/L, 在EC/DMC (1:1)中, 来源: 电池文献
    anion_size=3.2,  # Å, TFSI⁻半径, 来源: DFT计算 (Chem. Mater.)
    dissociation_constant=0.98,  # 无单位, 高解离近似值, 来源: 文献估计
    thermal_stability=300,  # °C, 分解温度, 来源: 实验值 (Electrochim. Acta)
    electrochemical_stability=5.0,  # V vs. Li/Li⁺, 来源: 电池文献
)


# LithiumSalt.save_to_json("data/materials/lithiumsalt.json")
# Additive.save_to_json("data/materials/additive.json")
# Solvent.save_to_json("data/materials/solvent.json")
pass
