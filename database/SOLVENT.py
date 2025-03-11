from models import SolventModel, HydrogenBonding


EC = SolventModel(
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
    hydrogen_bonding=HydrogenBonding.WEAK,  # 羰基有一定氢键能力，但较弱
)

PC = SolventModel(
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
    hydrogen_bonding=HydrogenBonding.WEAK,  # 羰基存在弱氢键能力，来源: 化学性质分析
)

DMC = SolventModel(
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
    hydrogen_bonding=HydrogenBonding.NONE,  # 无强氢键给体或受体，来源: 分子结构分析
)

EMC = SolventModel(
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
    hydrogen_bonding=HydrogenBonding.NONE,  # 无强氢键能力，来源: 分子结构分析
)

DEC = SolventModel(
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
    hydrogen_bonding=HydrogenBonding.NONE,  # 无强氢键能力，来源: 分子结构分析
)
