from elml.dataset import ElectrolyteDataset
from elml.ml.models import ElectrolyteTransformer
from elml.ml.workflow import TrainingWorkflow

# 1. 准备数据
train_ds, val_ds, test_ds = ElectrolyteDataset.create_splits(
    "data/calisol23/calisol23.json", feature_mode="sequence"
)

# 2. 初始化模型
model = ElectrolyteTransformer(input_dim=179, model_dim=256, nhead=8)

# 3. 创建工作流
workflow = TrainingWorkflow(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
    batch_size=32,
    lr=1e-4,
)

# 4. 训练模型
workflow.train(num_epochs=100, early_stopping_patience=20)

# 5. 评估模型
workflow.load_checkpoint("best_model.pth")
test_loss = workflow.evaluate(workflow.test_loader)

# 6. 绘制训练曲线
workflow.plot_training_history()
