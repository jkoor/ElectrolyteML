import argparse
import sys
import os

# 将项目根目录添加到Python路径中
# 这使得我们可以直接从 elml 导入模块，而无需复杂的相对导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from elml.dataset import ElectrolyteDataset
from elml.ml.models import ElectrolyteMLP, ElectrolyteTransformer
from elml.ml.workflow import TrainingWorkflow


def main():
    """
    主函数，用于解析命令行参数并启动训练工作流。
    """
    parser = argparse.ArgumentParser(
        description="Train an electrolyte property prediction model."
    )

    # --- 数据集相关参数 ---
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to the JSON file containing the electrolyte formulas.",
    )
    parser.add_argument(
        "--target_metric",
        type=str,
        default="conductivity",
        help="The target metric to predict (e.g., 'conductivity').",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8, help="Fraction of data for training."
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1, help="Fraction of data for validation."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for data splitting."
    )

    # --- 模型相关参数 ---
    parser.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        choices=["mlp", "transformer"],
        help="Type of model to train.",
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="weighted_average",
        choices=["weighted_average", "sequence"],
        help="Feature generation mode for the dataset.",
    )

    # --- 训练过程相关参数 ---
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/exp1",
        help="Directory to save logs and model checkpoints.",
    )

    args = parser.parse_args()

    # --- 1. 加载和划分数据集 ---
    print("Loading and splitting dataset...")
    train_ds, val_ds, test_ds = ElectrolyteDataset.create_splits(
        dataset_file=args.dataset_file,
        feature_mode=args.feature_mode,
        target_metric=args.target_metric,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_seed=args.random_seed,
    )

    # --- 2. 初始化模型 ---
    print(f"Initializing model of type: {args.model_type}")
    if args.model_type == "mlp":
        # 对于MLP，输入维度是固定的
        model = ElectrolyteMLP(input_dim=train_ds.FEATURE_DIM)
    elif args.model_type == "transformer":
        # 对于Transformer，输入维度需要包含占比信息
        model = ElectrolyteTransformer(input_dim=train_ds.FEATURE_DIM + 1)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # --- 3. 初始化并运行工作流 ---
    print("Initializing training workflow...")
    workflow = TrainingWorkflow(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=args.batch_size,
        lr=args.lr,
        log_dir=args.log_dir,
    )

    # --- 4. 开始训练 ---
    workflow.train(num_epochs=args.num_epochs)

    # --- 5. 在测试集上评估最终模型 ---
    print("\nEvaluating on the test set with the best model...")
    workflow.load_checkpoint("best_model.pth")
    workflow.evaluate(workflow.test_loader)


if __name__ == "__main__":
    main()
