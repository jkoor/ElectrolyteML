from elml import MLibrary
from elml.battery import Electrolyte
from elml.dataset import ElectrolyteDataset

# --- 步骤 1: 初始化环境 ---
# 加载材料库 (由 material 模块自动完成)
print(f"✅ 1. 材料库已加载，共 {len(MLibrary)} 种材料。")

# --- 步骤 2: 创建一个新的电解液配方 ---
# 假设我们要设计一个高浓度电解液
my_new_recipe = Electrolyte.create(
    name="My-Custom-High-Concentration-Electrolyte",
    salts=[{"abbr": "LiPF6", "overall_fraction": 15.0}],  # 盐浓度提高到 15%
    solvents=[
        {"abbr": "EC", "relative_fraction": 40.0},
        {"abbr": "DMC", "relative_fraction": 60.0},
    ],
    additives=[{"abbr": "VC", "overall_fraction": 2.0}],  # 添加 2% 的 VC 添加剂
)
print(f"✅ 2. 已创建新配方: '{my_new_recipe.name}'")
my_new_recipe.show()


# --- 步骤 3: 加载数据集并训练模型 ---
# 使用 calisol23 数据集进行训练
dataset = ElectrolyteDataset("data/calisol23.json")

print("✅ 3. 开始训练性能预测模型...")
dataset.train_model(
    model="transformer",
    model_config={"hidden_dim": 256, "n_layers": 2, "n_heads": 4},
    epochs=100,  # 实际应用中可能需要更多轮次
    learning_rate=0.001,
    save_path="models/best_model.pth",
)
print("模型训练完毕！")


# --- 步骤 4: 预测新配方的性能 ---
# 使用刚刚训练好的模型进行预测
print(f"✅ 4. 预测新配方 '{my_new_recipe.name}' 的性能...")
predicted_performance = dataset.predict(my_new_recipe)

print("=" * 50)
print(f"🎉 预测结果 🎉")
print(f"配方: {my_new_recipe.name}")
print(f"预测的离子电导率: {predicted_performance:.4f} mS/cm")
print("=" * 50)
