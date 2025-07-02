import argparse
import torch
import json
import os
from torch.utils.data import DataLoader, random_split

# (其他 imports 保持不变)
from elml.dataset import ElectrolyteDataset
from elml.ml.models import ElectrolyteTransformer
from elml.ml.trainer import Trainer
from elml.ml.evaluator import Evaluator


def set_seed(seed):
    # ... (函数不变)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # ... (阶段 0: 设置与准备 - 不变) ...
    print(f">>> 实验开始: {args.exp_name} <<<")
    set_seed(args.seed)
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
    model_save_path = os.path.join(output_dir, "best_model.pth")
    plot_save_path = os.path.join(output_dir, "evaluation_plot.png")

    # === 1. 数据准备阶段 ===
    print("\n>>> 阶段 1: 加载和准备数据...")
    full_dataset = ElectrolyteDataset(dataset_file=args.data_file)
    print(f"完整数据集加载完成，共 {len(full_dataset)} 个样本。")

    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    print(
        f"数据已划分为: {len(train_dataset)} (训练), {len(val_dataset)} (验证), {len(test_dataset)} (测试)"
    )

    # --- 关键修改：为 DataLoader 添加 num_workers 和 pin_memory ---
    # 这会开启后台子进程并行加载数据，并使用锁页内存加速CPU到GPU的传输
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # --- 修改结束 ---

    # === 2. 模型训练阶段 ===
    # ... (此部分代码不变) ...
    print("\n>>> 阶段 2: 初始化模型并开始训练...")
    model = ElectrolyteTransformer(
        input_dim=full_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    trainer = Trainer(model=model)
    history = trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_path=model_save_path,
        warmup_epochs=10,
        lr_patience=args.lr_patience,  # <-- 传递新参数
        lr_factor=args.lr_factor,  # <-- 传递新参数
    )

    # === 3. 模型评估阶段 ===
    # ... (此部分代码不变) ...
    print("\n>>> 阶段 3: 加载最佳模型并在独立的测试集上进行最终评估...")
    evaluation_model = ElectrolyteTransformer(
        input_dim=full_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    evaluator = Evaluator(model=evaluation_model, model_path=model_save_path)
    metrics, targets, predictions = evaluator.evaluate(test_loader)
    evaluator.plot_predictions(targets, predictions, save_path=plot_save_path)
    with open(
        os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✅ 实验 '{args.exp_name}' 完成！结果已保存在 '{output_dir}' 目录中。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="电解液电导率预测模型 - 训练与评估脚本"
    )

    # ... (其他参数不变) ...
    parser.add_argument("--exp_name", type=str, default="exp1", help="为本次实验命名")
    parser.add_argument(
        "--data_file", type=str, default="data/calisol23.json", help="数据集文件路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs", help="保存所有实验结果的根目录"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于保证可复现性"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="最大训练周期数")
    parser.add_argument("--lr", type=float, default=0.0005, help="学习率")
    parser.add_argument("--patience", type=int, default=50, help="早停的耐心值")
    parser.add_argument(
        "--lr_patience", type=int, default=10, help="学习率调度器的耐心值"
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="学习率降低的乘法因子",
    )

    # --- 关键修改：将 batch_size 和 num_workers 作为可调参数 ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批次大小，尝试增大此值，如64, 128, 256",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载的子进程数量，可设为CPU核心数",
    )
    # --- 修改结束 ---

    # --- 模型超参数 ---
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Transformer模型隐藏层维度"
    )
    parser.add_argument("--n_layers", type=int, default=2, help="Transformer模型层数")
    parser.add_argument(
        "--n_heads", type=int, default=2, help="Transformer模型多头注意力头数"
    )
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout比例")

    args = parser.parse_args()
    main(args)

# uv run test.py --exp_name "muti_simple_model_with_warmup" --n_layers 2 --hidden_dim 128 --dropout 0.15 --num_workers 16 --batch_size 128
