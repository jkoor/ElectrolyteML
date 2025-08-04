"""
预测结果分析模块
用于对比预测值与实际值，生成评估指标和可视化图表
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

from ..dataset import ElectrolyteDataset
from .predictor import ElectrolytePredictor


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Arial", "DejaVu Sans"]


class ElectrolyteAnalyzer:
    """
    电解液预测结果分析器

    提供全面的预测结果分析功能：
    - 评估指标计算 (MAE, RMSE, R², MAPE)
    - 可视化图表生成
    - 详细样本分析
    - 误差分析
    - 统计检验
    """

    def __init__(
        self,
        predictor: ElectrolytePredictor,
        save_dir: str = "analysis_results",
        figure_size: Tuple[int, int] = (15, 10),
        dpi: int = 300,
        font_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化分析器

        Args:
            predictor: 训练好的预测器实例
            save_dir: 分析结果保存目录
            figure_size: 图表尺寸 (宽, 高)
            dpi: 图表分辨率
            font_config: 字体配置字典
        """
        self.predictor = predictor
        self.save_dir = save_dir
        self.figure_size = figure_size
        self.dpi = dpi

        # 配置字体
        self._configure_font(font_config)

        # 存储分析结果
        self.results = {}
        self.predictions = None
        self.actual_values = None

        print("ElectrolyteAnalyzer initialized")
        print(f"Save directory: {self.save_dir}")

    def _configure_font(self, font_config: Optional[Dict[str, Any]] = None):
        """配置matplotlib字体"""
        default_config = {
            "font.sans-serif": ["Noto Sans CJK SC", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
        }

        config = font_config or default_config
        for key, value in config.items():
            plt.rcParams[key] = value

    def analyze_dataset(
        self,
        dataset: ElectrolyteDataset,
        show_plots: bool = False,
        save_plots: bool = True,
        plot_filename: Optional[str] = None,
        return_original_scale: bool = True,
    ) -> Dict[str, Any]:
        """
        分析数据集的预测结果

        Args:
            dataset: 要分析的数据集
            show_plots: 是否显示图表
            save_plots: 是否保存图表
            plot_filename: 图表保存文件名
            return_original_scale: 是否返回原始尺度的结果

        Returns:
            包含所有分析结果的字典
        """
        print("=== 开始预测结果分析 ===")

        # 进行预测并获取实际值
        self.predictions, self.actual_values = self.predictor.predict_dataset(
            dataset, return_original_scale=return_original_scale
        )

        # 计算基本统计信息
        self._calculate_basic_statistics()

        # 计算评估指标
        self._calculate_metrics()

        # 创建可视化图表
        if save_plots or show_plots:
            filename = plot_filename or f"{self.save_dir}/prediction_analysis.png"
            self._create_analysis_plots(show_plots, save_plots, filename)

        # 详细样本分析
        self._detailed_sample_analysis()

        # 误差分析
        self._error_analysis()

        # 统计检验
        self._statistical_tests()

        print("=== 分析完成 ===")
        return self.results

    def _calculate_basic_statistics(self):
        """计算基本统计信息"""
        pred_np = self.predictions
        actual_np = self.actual_values

        if pred_np is None or actual_np is None:
            raise ValueError("Predictions or actual values are not available.")

        print(f"测试样本数量: {len(pred_np)}")
        print(f"预测值范围: {pred_np.min():.4f} - {pred_np.max():.4f}")
        print(f"实际值范围: {actual_np.min():.4f} - {actual_np.max():.4f}")
        print(f"预测值均值: {pred_np.mean():.4f}")
        print(f"实际值均值: {actual_np.mean():.4f}")

        self.results.update(
            {
                "sample_count": len(pred_np),
                "pred_range": {
                    "min": float(pred_np.min()),
                    "max": float(pred_np.max()),
                },
                "actual_range": {
                    "min": float(actual_np.min()),
                    "max": float(actual_np.max()),
                },
                "pred_mean": float(pred_np.mean()),
                "actual_mean": float(actual_np.mean()),
                "pred_std": float(pred_np.std()),
                "actual_std": float(actual_np.std()),
            }
        )

    def _calculate_metrics(self):
        """计算评估指标"""
        pred_np = self.predictions
        actual_np = self.actual_values

        if pred_np is None or actual_np is None:
            raise ValueError("Predictions or actual values are not available.")

        # 计算评估指标
        mae = mean_absolute_error(actual_np, pred_np)
        mse = mean_squared_error(actual_np, pred_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_np, pred_np)

        # 计算平均绝对百分比误差(MAPE)
        # 避免除零错误
        non_zero_mask = actual_np != 0
        if non_zero_mask.any():
            mape = (
                np.mean(
                    np.abs(
                        (actual_np[non_zero_mask] - pred_np[non_zero_mask])
                        / actual_np[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            mape = float("inf")

        print("\n评估指标:")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")

        self.results.update(
            {
                "predictions": pred_np.tolist(),
                "actual": actual_np.tolist(),
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2),
                "mape": float(mape),
            }
        )

    def _create_analysis_plots(self, show_plots: bool, save_plots: bool, filename: str):
        """创建分析图表"""
        actual = self.actual_values
        predicted = self.predictions

        if actual is None or predicted is None:
            raise ValueError(
                "Predictions or actual values are not available for plotting."
            )

        r2 = self.results["r2"]
        mae = self.results["mae"]
        mape = self.results["mape"]

        plt.figure(figsize=self.figure_size)

        # 子图1: 预测值vs实际值散点图
        plt.subplot(2, 3, 1)
        plt.scatter(actual, predicted, alpha=0.6, s=30, color="blue")
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.title(f"Predicted vs Actual Value\n$R^2$ = {r2:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 添加拟合线
        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        plt.plot(
            actual,
            p(actual),
            "g--",
            alpha=0.8,
            label=f"Fit Line (y={z[0]:.3f}x+{z[1]:.3f})",
        )
        plt.legend()

        # 子图2: 残差图
        plt.subplot(2, 3, 2)
        residuals = predicted - actual
        plt.scatter(actual, residuals, alpha=0.6, s=30, color="orange")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Actual Value")
        plt.ylabel("Residuals (Predicted - Actual)")
        plt.title("Residuals Distribution Plot")
        plt.grid(True, alpha=0.3)

        # 子图3: 误差分布直方图
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor="black", color="green")
        plt.axvline(x=0, color="r", linestyle="--", label="Zero Error Line")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title(f"Residuals Histogram\nMAE = {mae:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图4: Q-Q图检验残差正态性
        plt.subplot(2, 3, 4)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Residuals Q-Q Plot\n(Normality Test)")
        plt.grid(True, alpha=0.3)

        # 子图5: 预测值与实际值的时间序列对比（前50个样本）
        plt.subplot(2, 3, 5)
        n_show = min(50, len(predicted))
        x_range = range(n_show)
        plt.plot(
            x_range,
            actual[:n_show],
            "o-",
            label="Actual Value",
            markersize=4,
            color="blue",
        )
        plt.plot(
            x_range,
            predicted[:n_show],
            "s-",
            label="Predicted Value",
            markersize=4,
            color="red",
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title(f"First {n_show} Samples Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图6: 误差分布的累积分布函数
        plt.subplot(2, 3, 6)
        abs_errors = np.abs(residuals)
        sorted_errors = np.sort(abs_errors)
        y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, y, "b-", linewidth=2)
        plt.xlabel("Absolute Error")
        plt.ylabel("Cumulative Probability")
        plt.title(f"Absolute Error CDF\nMAPE = {mape:.2f}%")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            import os

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"\n图表已保存为: {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _detailed_sample_analysis(self, n_samples: int = 10):
        """详细样本分析"""
        actual = self.actual_values
        predicted = self.predictions

        if actual is None or predicted is None:
            raise ValueError(
                "Predictions or actual values are not available for detailed analysis."
            )

        print(f"\n前{n_samples}个样本详细对比:")
        print("索引\t实际值\t\t预测值\t\t绝对误差\t相对误差(%)")
        print("-" * 70)

        sample_details = []
        for i in range(min(n_samples, len(predicted))):
            error = predicted[i] - actual[i]
            abs_error = abs(error)
            rel_error = (
                (abs_error / actual[i]) * 100 if actual[i] != 0 else float("inf")
            )

            sample_details.append(
                {
                    "index": int(i),
                    "actual": float(actual[i]),
                    "predicted": float(predicted[i]),
                    "absolute_error": float(abs_error),
                    "relative_error": float(rel_error),
                }
            )

            print(
                f"{i}\t{actual[i]:.6f}\t{predicted[i]:.6f}\t{abs_error:.6f}\t{rel_error:.2f}"
            )

        self.results["sample_details"] = sample_details

    def _error_analysis(self):
        """误差分析"""

        if self.predictions is None or self.actual_values is None:
            raise ValueError(
                "Predictions or actual values are not available for error analysis."
            )

        actual = self.actual_values
        predicted = self.predictions
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
        error_stats = {
            "mean_abs_error": float(np.mean(abs_errors)),
            "error_std": float(np.std(residuals)),
            "error_median": float(np.median(residuals)),
            "error_95_percentile": float(np.percentile(abs_errors, 95)),
            "best_prediction": {
                "index": int(best_idx),
                "actual": float(actual[best_idx]),
                "predicted": float(predicted[best_idx]),
                "absolute_error": float(abs_errors[best_idx]),
            },
            "worst_prediction": {
                "index": int(worst_idx),
                "actual": float(actual[worst_idx]),
                "predicted": float(predicted[worst_idx]),
                "absolute_error": float(abs_errors[worst_idx]),
            },
        }

        print("\n误差统计:")
        print(f"  平均绝对误差: {error_stats['mean_abs_error']:.6f}")
        print(f"  误差标准差: {error_stats['error_std']:.6f}")
        print(f"  误差中位数: {error_stats['error_median']:.6f}")
        print(f"  95%误差分位数: {error_stats['error_95_percentile']:.6f}")

        # 分析大误差样本
        high_error_threshold = np.percentile(abs_errors, 90)  # 90分位数
        high_error_indices = np.where(abs_errors > high_error_threshold)[0]

        print(f"\n高误差样本分析 (误差 > {high_error_threshold:.4f}):")
        print(
            f"  高误差样本数量: {len(high_error_indices)} ({len(high_error_indices) / len(actual) * 100:.1f}%)"
        )

        high_error_samples = []
        if len(high_error_indices) > 0:
            print("  前5个高误差样本:")
            sorted_high_error = high_error_indices[
                np.argsort(abs_errors[high_error_indices])[::-1]
            ]
            for i, idx in enumerate(sorted_high_error[:5]):
                sample_info = {
                    "index": int(idx),
                    "actual": float(actual[idx]),
                    "predicted": float(predicted[idx]),
                    "absolute_error": float(abs_errors[idx]),
                }
                high_error_samples.append(sample_info)
                print(
                    f"    样本{idx}: 实际={actual[idx]:.4f}, 预测={predicted[idx]:.4f}, 误差={abs_errors[idx]:.4f}"
                )

        error_stats["high_error_threshold"] = float(high_error_threshold)
        error_stats["high_error_samples"] = high_error_samples
        self.results["error_analysis"] = error_stats

    def _statistical_tests(self):
        """统计检验"""

        if self.predictions is None or self.actual_values is None:
            raise ValueError(
                "Predictions or actual values are not available for statistical tests."
            )

        residuals = self.predictions - self.actual_values

        # Shapiro-Wilk正态性检验
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        except Exception:
            shapiro_stat, shapiro_p = None, None

        # Kolmogorov-Smirnov正态性检验
        try:
            ks_stat, ks_p = stats.kstest(residuals, "norm")
        except Exception:
            ks_stat, ks_p = None, None

        # Durbin-Watson自相关检验（如果适用）
        # Durbin-Watson自相关检验（检测残差的序列相关性）
        try:
            diff = np.diff(residuals)
            dw_stat = np.sum(diff**2) / np.sum(residuals**2)
        except Exception:
            dw_stat = None

        statistical_tests = {
            "shapiro_wilk": {
                "statistic": float(shapiro_stat) if shapiro_stat is not None else None,
                "p_value": float(shapiro_p) if shapiro_p is not None else None,
                "is_normal": bool(shapiro_p > 0.05) if shapiro_p is not None else None,
            },
            "kolmogorov_smirnov": {
                "statistic": float(ks_stat) if ks_stat is not None else None,
                "p_value": float(ks_p) if ks_p is not None else None,
                "is_normal": bool(ks_p > 0.05) if ks_p is not None else None,
            },
            "durbin_watson": {
                "statistic": float(dw_stat) if dw_stat is not None else None,
                "autocorrelation": "positive"
                if dw_stat and dw_stat < 1.5
                else "negative"
                if dw_stat and dw_stat > 2.5
                else "none"
                if dw_stat
                else None,
            },
        }

        self.results["statistical_tests"] = statistical_tests

        print("\n=== 统计检验 ===")
        if shapiro_p is not None:
            print(
                f"Shapiro-Wilk正态性检验: p = {shapiro_p:.4f} ({'正态' if shapiro_p > 0.05 else '非正态'})"
            )
        if ks_p is not None:
            print(
                f"Kolmogorov-Smirnov检验: p = {ks_p:.4f} ({'正态' if ks_p > 0.05 else '非正态'})"
            )
        if dw_stat is not None:
            print(f"Durbin-Watson自相关检验: {dw_stat:.4f}")

    def compare_models(
        self,
        other_analyzer: "ElectrolyteAnalyzer",
        dataset: ElectrolyteDataset,
        model_names: Tuple[str, str] = ("Model 1", "Model 2"),
    ) -> Dict[str, Any]:
        """
        比较两个模型的性能

        Args:
            other_analyzer: 另一个分析器实例
            dataset: 测试数据集
            model_names: 模型名称元组

        Returns:
            模型比较结果
        """
        # 分析当前模型
        results1 = self.analyze_dataset(dataset, show_plots=False, save_plots=False)

        # 分析另一个模型
        results2 = other_analyzer.analyze_dataset(
            dataset, show_plots=False, save_plots=False
        )

        # 比较指标
        comparison = {
            "model_names": model_names,
            "metrics_comparison": {
                "mae": {
                    model_names[0]: results1["mae"],
                    model_names[1]: results2["mae"],
                    "improvement": (
                        (results2["mae"] - results1["mae"]) / results2["mae"]
                    )
                    * 100,
                },
                "rmse": {
                    model_names[0]: results1["rmse"],
                    model_names[1]: results2["rmse"],
                    "improvement": (
                        (results2["rmse"] - results1["rmse"]) / results2["rmse"]
                    )
                    * 100,
                },
                "r2": {
                    model_names[0]: results1["r2"],
                    model_names[1]: results2["r2"],
                    "improvement": (
                        (results1["r2"] - results2["r2"]) / abs(results2["r2"])
                    )
                    * 100,
                },
                "mape": {
                    model_names[0]: results1["mape"],
                    model_names[1]: results2["mape"],
                    "improvement": (
                        (results2["mape"] - results1["mape"]) / results2["mape"]
                    )
                    * 100,
                },
            },
        }

        print("\n=== 模型比较结果 ===")
        for metric, values in comparison["metrics_comparison"].items():
            print(f"{metric.upper()}:")
            print(f"  {model_names[0]}: {values[model_names[0]]:.4f}")
            print(f"  {model_names[1]}: {values[model_names[1]]:.4f}")
            print(f"  改进: {values['improvement']:+.2f}%")

        return comparison

    def save_results(self, filename: str):
        """保存分析结果到JSON文件"""
        import json
        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"分析结果已保存到: {filename}")

    def get_summary(self) -> Dict[str, Any]:
        """获取分析结果摘要"""
        if not self.results:
            return {
                "error": "No analysis results available. Run analyze_dataset first."
            }

        return {
            "sample_count": self.results.get("sample_count"),
            "metrics": {
                "mae": self.results.get("mae"),
                "rmse": self.results.get("rmse"),
                "r2": self.results.get("r2"),
                "mape": self.results.get("mape"),
            },
            "error_analysis": {
                "best_error": self.results.get("error_analysis", {})
                .get("best_prediction", {})
                .get("absolute_error"),
                "worst_error": self.results.get("error_analysis", {})
                .get("worst_prediction", {})
                .get("absolute_error"),
                "error_std": self.results.get("error_analysis", {}).get("error_std"),
            },
        }


# 为了向后兼容，保留原来的函数接口
def analyze_predictions(
    predictor: ElectrolytePredictor,
    test_dataset: ElectrolyteDataset,
    show_plots=False,
    save_plots=True,
    plot_filename="prediction_analysis.png",
) -> Dict[str, Any]:
    """
    分析预测结果的函数（向后兼容接口）

    Args:
        predictor: 训练好的预测器实例
        test_dataset: 测试数据集
        show_plots: 是否显示图表
        save_plots: 是否保存图表
        plot_filename: 图表保存文件名

    Returns:
        分析结果字典
    """
    analyzer = ElectrolyteAnalyzer(predictor)
    return analyzer.analyze_dataset(
        test_dataset,
        show_plots=show_plots,
        save_plots=save_plots,
        plot_filename=plot_filename,
    )
