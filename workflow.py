from elml.dataset import ElectrolyteDataset
from elml.ml import (
    ElectrolyteTransformer,
    ElectrolyteMLP,
    ElectrolytePredictor,
    ElectrolyteTrainer,
)
from elml import MLibrary
from elml.battery import Electrolyte


# 1. 准备数据
train_ds, val_ds, test_ds = ElectrolyteDataset.create_splits(
    "data/calisol23/calisol23.json", feature_mode="sequence"
)

# 2. 初始化模型
model = ElectrolyteTransformer(input_dim=179, model_dim=256, nhead=8)

# 3. 创建工作流
trainer = ElectrolyteTrainer(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
    batch_size=32,
    lr=1e-4,
)

predictor = ElectrolytePredictor(
    model=model,
    model_checkpoint_path="runs/transformer/best_model.pth",
    feature_mode="sequence",
)

lipf6 = MLibrary.get_material(abbr="LiPF6")
ec = MLibrary.get_material(abbr="EC")
emc = MLibrary.get_material(abbr="EMC")
dmc = MLibrary.get_material(abbr="DMC")
vc = MLibrary.get_material(abbr="VC")

lp30_recipe = Electrolyte.create(
    name="LP30",
    id="standard-lp30",
    description="一个标准的 LiPF6 在 EC/DMC 中的电解液配方",
    salts=[(lipf6, 12.5)],  #  对于盐，overall_fraction 是其在总溶剂质量中的重量百分比
    solvents=[
        (ec, 33.3),
        (emc, 33.3),
        (dmc, 33.4),  # DMC 在此配方中不使用
    ],  # 对于溶剂，relative_fraction 是其在所有溶剂中的相对体积或重量比
    additives=[],  # 也可以添加添加剂
    performance={"temperature": 298.15},  # 测试条件
)

result = predictor.predict_single_formula(
    electrolyte=lp30_recipe,
    temperature=298.15,
    return_original_scale=True,
)
print(f"Conductivity at 298.15 K: {result:.4f} S/m")
# 4. 训练模型
# trainer.train(num_epochs=100, early_stopping_patience=20)

# 5. 评估模型
# trainer.load_checkpoint("best_model.pth")
# test_loss = trainer.evaluate(trainer.test_loader)

# 6. 绘制训练曲线
# trainer.plot_training_history()

# 7. 进行预测结果分析
