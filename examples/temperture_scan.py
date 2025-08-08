from elml import MLibrary
from elml.battery import Electrolyte
from elml.ml import ElectrolyteWorkflow, WorkflowConfig
import numpy as np
from typing import Dict

config = WorkflowConfig(
    input_dim=179,
    model_type="transformer",
    data_path="data/calisol23/calisol23.json",
    model_name="transformer_model_12_splits_custom_25",
)

workflow = ElectrolyteWorkflow(config)
# 仅执行预测和分析部分
workflow.setup_data()
workflow.setup_model()
workflow.setup_predictor(str(workflow.log_dir / "best_model.pth"))


def predict_conductivity(electrolyte: Electrolyte) -> float:
    """Predict the conductivity of an electrolyte."""

    if not workflow.predictor:
        raise ValueError("Predictor is not set up correctly.")
    result: float = workflow.predictor.predict_single_formula(electrolyte)
    return result


def predict_conductivity_over_temperature_range(
    electrolyte: Electrolyte, start_temp: float, end_temp: float, step: float
) -> Dict[float, float]:
    """
    在给定的温度范围内预测电解质的电导率。

    Args:
        electrolyte: 电解质对象。
        start_temp: 起始温度（摄氏度）。
        end_temp: 结束温度（摄氏度）。
        step: 温度扫描间隔（摄氏度）。

    Returns:
        一个字典，键是温度，值是预测的电导率。
    """

    # 将摄氏度转换为开尔文
    start_temp += 273.15
    end_temp += 273.15

    results = {}
    for temp in np.arange(start_temp, end_temp + step, step):
        # 创建电解质的深层副本以避免修改原始对象
        electrolyte_copy = Electrolyte.from_dict(electrolyte.to_dict())
        electrolyte_copy.condition["temperature"] = temp
        conductivity = predict_conductivity(electrolyte_copy)
        results[temp] = conductivity
    return results


# 使用 create 方法定义一个复杂的电解液配方
ec_emc_3_7 = Electrolyte.create(
    id="complex-electrolyte-1",
    description="A complex electrolyte formulation with multiple components.",
    name="Complex-Recipe-1",
    salts=[(MLibrary.get_material("LiPF6"), 15.0)],
    solvents=[
        (MLibrary.get_material("EC"), 30.0),
        (MLibrary.get_material("EMC"), 70.0),
    ],
    additives=[],
    performance={},
)

result = predict_conductivity_over_temperature_range(
    ec_emc_3_7, start_temp=-60, end_temp=60, step=10.0
)
print("Conductivity predictions over temperature range:")
for temp, cond in result.items():
    print(f"温度: {temp:.2f} °C, 电导率: {cond:.4f} S/m")
