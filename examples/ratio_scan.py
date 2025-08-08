from elml import MLibrary
from elml.battery import Electrolyte
from elml.material import Material
from elml.ml import ElectrolyteWorkflow, WorkflowConfig
from typing import Optional, Dict, Tuple, List
import itertools

# --- Workflow Setup ---
# This part sets up the machine learning model for prediction.
# It's configured once and reused for all predictions.
config = WorkflowConfig(
    input_dim=179,
    model_type="transformer",
    data_path="data/calisol23/calisol23.json",
    model_name="transformer_model_12_splits_custom_25",
)

workflow = ElectrolyteWorkflow(config)
print("Setting up workflow... (This may take a moment)")
workflow.setup_data()
workflow.setup_model()
workflow.setup_predictor(str(workflow.log_dir / "best_model.pth"))
print("Workflow setup complete.")


def predict_conductivity(electrolyte: Electrolyte) -> float:
    """
    Predict the conductivity of a single electrolyte formulation.
    Uses the pre-configured global workflow.
    """
    if not workflow.predictor:
        raise ValueError("Predictor is not set up correctly.")
    result: float = workflow.predictor.predict_single_formula(electrolyte)
    return result


def find_optimal_ratios(
    salt: Tuple[Material, float],
    solvents: List[Material],
    fixed_solvents: Dict[str, float] = None,
    step: float = 5.0,
) -> List[Tuple[Dict[str, float], float]]:
    """
    在不同组分比例下寻找最优电解液配方。

    Args:
        salt (Tuple[Material, float]): 盐及其浓度 (e.g., (MLibrary.get_material("LiPF6"), 15.0))
        solvents (List[Material]): 用于扫描比例的溶剂材料列表。
        fixed_solvents (Dict[str, float], optional): 一个字典，包含名称及其固定比例的溶剂。 Defaults to None.
        step (float, optional): 比例扫描的步长 (e.g., 5 for 5%)。 Defaults to 5.0.

    Returns:
        List[Tuple[Dict[str, float], float]]: 返回一个列表，包含前3个最优配方。
                                              每个元组包含一个配方字典和其预测的电导率。
    """
    if fixed_solvents is None:
        fixed_solvents = {}

    variable_solvents = [s for s in solvents if s.name not in fixed_solvents]
    if not variable_solvents:
        raise ValueError("至少需要一种可变比例的溶剂。")

    fixed_ratio_sum = sum(fixed_solvents.values())
    if fixed_ratio_sum >= 100.0:
        raise ValueError("固定比例的总和必须小于100。")

    remaining_ratio = 100.0 - fixed_ratio_sum
    num_variable = len(variable_solvents)
    results = []

    # 生成所有可能的比例组合
    num_steps = int(remaining_ratio / step)
    for p in itertools.product(range(num_steps + 1), repeat=num_variable):
        if sum(p) == num_steps and 0 not in p:
            current_ratios = {
                s.abbreviation: val * step for s, val in zip(variable_solvents, p)
            }
            current_ratios.update(fixed_solvents)

            # 创建电解质对象
            solvent_tuples = [
                (MLibrary.get_material(name), ratio)
                for name, ratio in current_ratios.items()
            ]
            electrolyte = Electrolyte.create(
                id=f"scan_{len(results)}",
                name="ScanRecipe",
                description="A scanned electrolyte formulation.",
                salts=[salt],
                solvents=solvent_tuples,
                additives=[],
                performance={},
            )

            # 预测电导率
            conductivity = predict_conductivity(electrolyte)
            results.append((current_ratios, conductivity))
            # print(f"测试配方: {current_ratios}, 预测电导率: {conductivity:.4f}")

    # 排序并返回前3名
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]


if __name__ == "__main__":
    # --- 示例用法 ---

    # 定义盐和要扫描的溶剂
    scan_salt = (MLibrary.get_material("LiPF6"), 15.0)
    scan_solvents = [
        MLibrary.get_material("EC"),
        MLibrary.get_material("EMC"),
        MLibrary.get_material("DMC"),
    ]

    # 1. 无固定组分，在 EC, EMC, DMC 中寻找最优配比
    print("\n--- 开始扫描 (无固定组分) ---")
    top_3_recipes = find_optimal_ratios(
        salt=scan_salt,
        solvents=scan_solvents,
        step=10,  # 使用10%的步长以加快速度
    )
    print("\n--- 无固定组分最优配方 Top 3 ---")
    for recipe, conductivity in top_3_recipes:
        print(f"配方: {recipe}, 预测电导率: {conductivity:.4f}")

    # 2. 固定 EC 的比例为 20%，在 EMC, DMC 中寻找最优配比
    print("\n--- 开始扫描 (固定EC=20%) ---")
    top_3_fixed_recipes = find_optimal_ratios(
        salt=scan_salt,
        solvents=scan_solvents[1:],
        fixed_solvents={"EC": 20.0},
        step=10,
    )
    print("\n--- 固定EC=20%最优配方 Top 3 ---")
    for recipe, conductivity in top_3_fixed_recipes:
        print(f"配方: {recipe}, 预测电导率: {conductivity:.4f}")
