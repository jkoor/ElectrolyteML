from models import LithiumSaltModel

LIPF6 = LithiumSaltModel(
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

LIFSI = LithiumSaltModel(
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

LITFSI = LithiumSaltModel(
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
