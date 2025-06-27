from dataset import ElectrolyteDataset
from ml import ElectrolyteTransformer, ElectrolyteMLP
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

# eds = ElectrolyteDataset("data/calisol23.json")

# 初始化数据集
dataset = ElectrolyteDataset("data/calisol23.json")

# 训练 Transformer
history = dataset.train_model(
    model="transformer",
    model_config={"hidden_dim": 256, "n_layers": 2, "n_heads": 2},
    train_ratio=0.8,
    batch_size=32,
    epochs=100,
    lr=0.001,
    patience=10,
    save_path="transformer_best.pth",
)

# 绘制损失曲线
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# 绘制 R² 曲线
plt.plot(history["val_r2"], label="Val R²")
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.legend()
plt.title("Validation R² Score")
plt.show()

# 预测候选配方
model = ElectrolyteTransformer(input_dim=178)
model.load_state_dict(torch.load("transformer_best.pth"))
model.eval()
predictions = dataset.predict_candidates(model)
for formula, pred in predictions[:5]:
    print(f"配方 {formula._data.id}: 预测电导率 {pred:.2f} mS/cm")

# 可视化验证集预测（需要修改 evaluate_model 返回预测值）
val_loader = DataLoader(
    TensorDataset(dataset.features, dataset.targets)[int(0.8 * len(dataset)) :],
    batch_size=32,
)
metrics = dataset.evaluate_model(
    model, val_loader, device="cuda" if torch.cuda.is_available() else "cpu"
)
print("验证集评估指标：")
for key, value in metrics.items():
    print(f"{key.upper()}: {value:.4f}")
