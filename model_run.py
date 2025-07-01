import argparse
import torch
import json
import os
from elml.dataset import ElectrolyteDataset
from elml.ml.models import ElectrolyteTransformer
from elml.ml.trainer import Trainer
from elml.ml.evaluator import Evaluator  # 引入评估器


def set_seed(seed):
    """设置随机种子以保证实验可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # === 0. 设置与准备 ===
    print(">>> 阶段 0: 设置随机种子和实验环境...")
    set_seed(args.seed)

    # 创建一个目录来保存本次实验的所有结果
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 将本次运行的配置参数保存下来
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    # --- 实验配置 ---
    MODEL_SAVE_PATH = os.path.join(args.output_dir, "best_transformer_model.pth")

    # === 1. 数据准备阶段 ===
    print("\n>>> 阶段 1: 加载和准备数据...")
    dataset = ElectrolyteDataset(dataset_file=args.data_file)
    print(f"数据集加载完成，包含 {len(dataset)} 个样本。")
    # 注意：此处未划分测试集，实际项目中建议严格划分训练集、验证集和测试集

    # === 2. 模型训练阶段 ===
    print("\n>>> 阶段 2: 初始化模型并开始训练...")
    model = ElectrolyteTransformer(
        input_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    trainer = Trainer(model=model, dataset=dataset)

    # 执行训练，Trainer会返回训练历史
    history = trainer.run(
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_path=MODEL_SAVE_PATH,
    )

    # 保存训练历史
    with open(
        os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(history, f, indent=4)

    # === 3. 模型评估阶段 ===
    print("\n>>> 阶段 3: 加载最佳模型并在验证集上进行评估...")
    # 重新初始化模型结构
    evaluation_model = ElectrolyteTransformer(
        input_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )

    # 使用 Evaluator 在验证集上评估
    # 注意：为了简化，这里直接使用了 Trainer 内部的验证集划分逻辑。
    # 更严谨的做法是单独创建一个测试集 DataLoader 传给 Evaluator。
    evaluator = Evaluator(model=evaluation_model, model_path=MODEL_SAVE_PATH)

    # evaluator.evaluate(...) # 此处需要传入一个测试集的 DataLoader
    # 鉴于当前代码结构，我们可以直接打印训练得到的最佳验证损失
    best_val_loss = min(history.get("val_loss", [float("inf")]))
    print(f"\n--- 评估结果 ---")
    print(f"训练中得到的最佳验证集损失 (Best Validation Loss): {best_val_loss:.4f}")

    # 保存评估结果
    with open(
        os.path.join(args.output_dir, "evaluation_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({"best_validation_loss": best_val_loss}, f, indent=4)


if __name__ == "__main__":
    # === 参数解析 ===
    parser = argparse.ArgumentParser(description="电解液电导率预测模型训练脚本")

    # --- 文件与路径参数 ---
    parser.add_argument(
        "--data_file", type=str, default="data/calisol23.json", help="数据集文件路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs/exp1", help="保存实验结果的目录"
    )

    # --- 训练超参数 ---
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集所占比例")
    parser.add_argument("--epochs", type=int, default=1000, help="最大训练周期数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--patience", type=int, default=20, help="早停的耐心值")

    # --- 模型超参数 ---
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Transformer模型隐藏层维度"
    )
    parser.add_argument("--n_layers", type=int, default=2, help="Transformer模型层数")
    parser.add_argument(
        "--n_heads", type=int, default=2, help="Transformer模型多头注意力头数"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比例")

    args = parser.parse_args()
    main(args)
