"""
预测结果分析脚本
用于对比预测值与实际值，生成评估指标和可视化图表
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from elml.dataset import ElectrolyteDataset
from elml.ml.models import ElectrolyteTransformer
from elml.ml.workflow import TrainingWorkflow


def analyze_predictions(
    workflow, test_dataset, save_plots=True, plot_filename="prediction_analysis.png"
):
    """
    分析预测结果的函数

    Args:
        workflow: 训练好的工作流实例
        test_dataset: 测试数据集
        save_plots: 是否保存图表
        plot_filename: 图表保存文件名
    """

    print("=== 开始预测结果分析 ===")

    # 进行预测并获取实际值
    predictions, actual_values = workflow.predict_dataset(
        test_dataset, return_targets=True
    )

    # 转换为numpy数组
    pred_np = predictions.detach().numpy().flatten()
    actual_np = actual_values.detach().numpy().flatten()

    # 由于数据可能做了log1p变换，需要还原到原始尺度
    pred_original = np.expm1(pred_np)  # 反log1p变换
    actual_original = np.expm1(actual_np)

    # 基本统计信息
    print(f"测试样本数量: {len(pred_original)}")
    print(f"预测值范围: {pred_original.min():.4f} - {pred_original.max():.4f}")
    print(f"实际值范围: {actual_original.min():.4f} - {actual_original.max():.4f}")
    print(f"预测值均值: {pred_original.mean():.4f}")
    print(f"实际值均值: {actual_original.mean():.4f}")

    # 计算评估指标
    mae = mean_absolute_error(actual_original, pred_original)
    mse = mean_squared_error(actual_original, pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_original, pred_original)

    # 计算平均绝对百分比误差(MAPE)
    mape = np.mean(np.abs((actual_original - pred_original) / actual_original)) * 100

    print("\n评估指标:")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R^2): {r2:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")

    # 创建可视化图表
    create_analysis_plots(
        actual_original, pred_original, r2, mae, mape, save_plots, plot_filename
    )

    # 详细样本分析
    detailed_sample_analysis(actual_original, pred_original)

    # 误差分析
    error_analysis(actual_original, pred_original)

    return {
        "predictions": pred_original,
        "actual": actual_original,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def create_analysis_plots(actual, predicted, r2, mae, mape, save_plots, filename):
    """创建分析图表"""

    plt.figure(figsize=(15, 10))
    # 显示中文
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
    # 显示负号
    plt.rcParams["axes.unicode_minus"] = False
    # 启用 LaTeX 渲染以支持数学符号
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.default"] = "regular"

    # 子图1: 预测值vs实际值散点图
    plt.subplot(2, 3, 1)
    plt.scatter(actual, predicted, alpha=0.6, s=30, color="blue")
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction"
    )
    plt.xlabel("实际值")
    plt.ylabel("预测值")
    plt.title(f"预测值 vs 实际值\n$R^2$ = {r2:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加拟合线
    z = np.polyfit(actual, predicted, 1)
    p = np.poly1d(z)
    plt.plot(
        actual, p(actual), "g--", alpha=0.8, label=f"拟合线 (y={z[0]:.3f}x+{z[1]:.3f})"
    )
    plt.legend()

    # 子图2: 残差图
    plt.subplot(2, 3, 2)
    residuals = predicted - actual
    plt.scatter(actual, residuals, alpha=0.6, s=30, color="orange")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("实际值")
    plt.ylabel("残差 (预测值 - 实际值)")
    plt.title("残差分布图")
    plt.grid(True, alpha=0.3)

    # 子图3: 误差分布直方图
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor="black", color="green")
    plt.axvline(x=0, color="r", linestyle="--", label="零误差线")
    plt.xlabel("残差")
    plt.ylabel("频次")
    plt.title(f"残差分布直方图\nMAE = {mae:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图4: Q-Q图检验残差正态性
    from scipy import stats

    plt.subplot(2, 3, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("残差Q-Q图\n(检验正态性)")
    plt.grid(True, alpha=0.3)

    # 子图5: 预测值与实际值的时间序列对比（前50个样本）
    plt.subplot(2, 3, 5)
    n_show = min(50, len(predicted))
    x_range = range(n_show)
    plt.plot(x_range, actual[:n_show], "o-", label="实际值", markersize=4, color="blue")
    plt.plot(
        x_range, predicted[:n_show], "s-", label="预测值", markersize=4, color="red"
    )
    plt.xlabel("样本索引")
    plt.ylabel("值")
    plt.title(f"前{n_show}个样本对比")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图6: 误差分布的累积分布函数
    plt.subplot(2, 3, 6)
    abs_errors = np.abs(residuals)
    sorted_errors = np.sort(abs_errors)
    y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, y, "b-", linewidth=2)
    plt.xlabel("绝对误差")
    plt.ylabel("累积概率")
    plt.title(f"绝对误差累积分布\nMAPE = {mape:.2f}%")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\n图表已保存为: {filename}")

    plt.show()


def detailed_sample_analysis(actual, predicted, n_samples=10):
    """详细样本分析"""

    print(f"\n前{n_samples}个样本详细对比:")
    print("索引\t实际值\t\t预测值\t\t绝对误差\t相对误差(%)")
    print("-" * 70)

    for i in range(min(n_samples, len(predicted))):
        error = predicted[i] - actual[i]
        abs_error = abs(error)
        rel_error = (abs_error / actual[i]) * 100 if actual[i] != 0 else float("inf")
        print(
            f"{i}\t{actual[i]:.6f}\t{predicted[i]:.6f}\t{abs_error:.6f}\t{rel_error:.2f}"
        )


def error_analysis(actual, predicted):
    """误差分析"""

    residuals = predicted - actual
    abs_errors = np.abs(residuals)

    # 找出最好和最差的预测
    best_idx = np.argmin(abs_errors)
    worst_idx = np.argmax(abs_errors)

    print("\n=== 误差分析 ===")
    print(f"最佳预测 (样本 {best_idx}):")
    print(f"  实际值: {actual[best_idx]:.6f}")
    print(f"  预测值: {predicted[best_idx]:.6f}")
    print(f"  绝对误差: {abs_errors[best_idx]:.6f}")
    print(f"  相对误差: {(abs_errors[best_idx] / actual[best_idx] * 100):.2f}%")

    print(f"\n最差预测 (样本 {worst_idx}):")
    print(f"  实际值: {actual[worst_idx]:.6f}")
    print(f"  预测值: {predicted[worst_idx]:.6f}")
    print(f"  绝对误差: {abs_errors[worst_idx]:.6f}")
    print(f"  相对误差: {(abs_errors[worst_idx] / actual[worst_idx] * 100):.2f}%")

    # 误差统计
    print("\n误差统计:")
    print(f"  平均绝对误差: {np.mean(abs_errors):.6f}")
    print(f"  误差标准差: {np.std(residuals):.6f}")
    print(f"  误差中位数: {np.median(residuals):.6f}")
    print(f"  95%误差分位数: {np.percentile(abs_errors, 95):.6f}")

    # 分析大误差样本
    high_error_threshold = np.percentile(abs_errors, 90)  # 90分位数
    high_error_indices = np.where(abs_errors > high_error_threshold)[0]

    print(f"\n高误差样本分析 (误差 > {high_error_threshold:.4f}):")
    print(
        f"  高误差样本数量: {len(high_error_indices)} ({len(high_error_indices) / len(actual) * 100:.1f}%)"
    )

    if len(high_error_indices) > 0:
        print("  前5个高误差样本:")
        sorted_high_error = high_error_indices[
            np.argsort(abs_errors[high_error_indices])[::-1]
        ]
        for i, idx in enumerate(sorted_high_error[:5]):
            print(
                f"    样本{idx}: 实际={actual[idx]:.4f}, 预测={predicted[idx]:.4f}, 误差={abs_errors[idx]:.4f}"
            )


if __name__ == "__main__":
    # 如果直接运行此脚本，则执行完整的分析流程

    print("加载数据和模型...")

    # 准备数据
    train_ds, val_ds, test_ds = ElectrolyteDataset.create_splits(
        "data/calisol23/calisol23.json", feature_mode="sequence"
    )

    # 初始化模型
    model = ElectrolyteTransformer(input_dim=179, model_dim=256, nhead=8)

    # 创建工作流
    workflow = TrainingWorkflow(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=32,
        lr=1e-4,
    )

    # 加载预训练模型
    try:
        workflow.load_checkpoint("best_model.pth")
        print("成功加载预训练模型")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请先训练模型或检查模型文件路径")
        exit(1)

    # 执行分析
    results = analyze_predictions(
        workflow,
        test_ds,
        save_plots=True,
        plot_filename="detailed_prediction_analysis.png",
    )

    print("\n=== 分析完成 ===")
    print(f"R^2 Score: {results['r2']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAPE: {results['mape']:.2f}%")
