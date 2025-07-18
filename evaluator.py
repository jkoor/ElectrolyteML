# run_evaluation.py

import torch
from torch.utils.data import TensorDataset, DataLoader

from elml.dataset import ElectrolyteDataset
from elml.ml.models import ElectrolyteTransformer  # 或 ElectrolyteMLP
from elml.ml.evaluator import Evaluator


def main():
    # --- 1. 参数配置 ---
    # !! 请确保这里的参数与您训练时使用的模型参数完全一致 !!
    TEST_DATASET_PATH = "data/calisol23.json"  # 指向您的独立测试数据集
    MODEL_PATH = "runs/exp1/best_transformer_model.pth"  # 指向您训练好的模型文件
    BATCH_SIZE = 32
    INPUT_DIM = 178  # 特征维度
    N_HEADS = 2  # Transformer模型的头数 (如果使用Transformer)
    SEQ_LEN = 5  # 序列最大长度 (如果使用MLP)

    # --- 2. 加载测试数据集 ---
    print(f"正在从 {TEST_DATASET_PATH} 加载测试数据...")
    # build_features=True 会自动处理数据并生成特征和标签
    test_dataset = ElectrolyteDataset(
        dataset_file=TEST_DATASET_PATH, build_features=True
    )

    if (
        len(test_dataset) == 0
        or test_dataset.targets is None
        or len(test_dataset.targets) == 0
    ):
        print("错误：测试数据集中没有可用于评估的带标签数据。")
        return

    test_tensor_dataset = TensorDataset(test_dataset.features, test_dataset.targets)
    test_loader = DataLoader(test_tensor_dataset, batch_size=BATCH_SIZE)

    # --- 3. 初始化模型和评估器 ---
    print(f"正在初始化模型...")
    # !! 关键：必须使用与训练时完全相同的模型结构和参数 !!
    model = ElectrolyteTransformer(input_dim=INPUT_DIM, n_heads=N_HEADS)
    # 如果您训练的是MLP模型，请使用下面这行:
    # model = ElectrolyteMLP(input_dim=INPUT_DIM, seq_len=SEQ_LEN)

    evaluator = Evaluator(model=model, model_path=MODEL_PATH)

    # --- 4. 执行评估 ---
    # evaluate 方法会返回指标字典、真实值和预测值
    metrics, targets, predictions = evaluator.evaluate(test_loader)

    # --- 5. 可视化评估结果 ---
    # 使用 plot_predictions 方法生成散点图，并保存
    evaluator.plot_predictions(
        targets, predictions, save_path="evaluation_results_plot.png"
    )


if __name__ == "__main__":
    main()
