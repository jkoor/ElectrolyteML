import itertools
import json
import random
from typing import Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import torch
from torch.utils.data import Dataset

from ..battery import Electrolyte
from ..material import MaterialLibrary
from .. import MLibrary


FeatureMode = Literal["sequence", "weighted_average"]


class ElectrolyteDataset(Dataset):
    """
    电解液数据集管理器。
    核心职责：加载、管理电解液配方数据，提供数据访问接口。
    与具体机器学习模型解耦，专注于数据管理。
    """

    def __init__(
        self,
        dataset_file: Optional[str] = None,
        feature_mode: FeatureMode = "sequence",
        max_components: int = 10,
        target_metric: str = "conductivity",
    ):
        """
        初始化 Dataset 类。

        Args:
            dataset_file (str, optional): 配方 JSON 文件路径。
            feature_mode (FeatureMode): 特征模式，默认为 "sequence"。
            max_components (int): 数据集中可能出现的最大组分数量，默认为 10。
            target_metric (str): 目标性能指标，默认为 "conductivity"。
        """

        # PyTorch Dataset 的构造函数调用
        super().__init__()

        self.library: MaterialLibrary = MLibrary
        self.formulas: list[Electrolyte] = []
        self.electrolyte_counter = itertools.count(1)
        self.target_metric = target_metric
        self.feature_mode: FeatureMode = feature_mode
        self.MAX_COMPONENTS = max_components
        self.FEATURE_DIM = (
            3 + 167 + 8 + 1
        )  # 材料类型(3), 分子指纹(167), 物化性质(8), 占比-Transformer序列模型/温度-简单模型(1) = 179

        # 温度归一化参数 (单位：K)
        # 针对锂电池电解液的实用温度范围
        self.TEMP_MIN = 223.15  # -50°C（低温性能测试）
        self.TEMP_MAX = 373.15  # 100°C（高温性能测试）

        if dataset_file:
            self.from_json(dataset_file)

    # ------------------------ 魔术方法 ------------------------ #
    def __len__(self) -> int:
        """返回数据集的样本总数"""
        return len(self.formulas)

    # 在 ElectrolyteDataset 类内部
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使数据集支持下标访问。
        为模型训练提供一个样本，包括特征和目标。
        """
        # 1. 根据索引获取原始的电解液配方对象
        electrolyte: Electrolyte = self.formulas[idx]
        return self._generate_all_tensor(electrolyte)

    def __iter__(self):
        """支持迭代访问"""
        return iter(self.formulas)

    # ------------------------ 数据管理方法 ------------------------ #

    def from_json(self, file_path: str):
        """从 JSON 文件加载配方数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            formulas_data: list[dict] = json.load(f)

        for data in formulas_data:
            self.add_formula(Electrolyte.from_dict(data))

    def to_json(self, file_path: str):
        """保存配方数据到 JSON 文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([f.to_dict() for f in self.formulas], f, indent=4)

    def add_formula(self, electrolyte: Electrolyte):
        """添加新配方"""
        electrolyte._data.id = f"E{next(self.electrolyte_counter)}"
        self.formulas.append(electrolyte)

    def remove_formula(self, formula_id: str):
        """删除指定 ID 的配方"""
        self.formulas = [f for f in self.formulas if f.id != formula_id]

    def get_formula_by_id(self, formula_id: str) -> Optional[Electrolyte]:
        """根据ID获取配方"""
        for formula in self.formulas:
            if formula.id == formula_id:
                return formula
        return None

    def update_performance(self, formula_id: str, performance_data: dict):
        """更新指定配方的性能数据"""
        formula = self.get_formula_by_id(formula_id)
        if formula:
            formula.performance.update(performance_data)

    # ------------------------ 数据查询方法 ------------------------ #

    def get_formulas_with_performance(
        self, metric: str = "conductivity"
    ) -> list[Electrolyte]:
        """
        获取包含指定性能指标的配方列表。
        """
        return [
            f
            for f in self.formulas
            if metric in f.performance and f.performance[metric] is not None
        ]

    def get_formulas_without_performance(
        self, metric: str = "conductivity"
    ) -> list[Electrolyte]:
        """
        获取不包含指定性能指标的配方列表（待预测样本）。
        """
        return [
            f
            for f in self.formulas
            if metric not in f.performance or f.performance[metric] is None
        ]

    def filter_formulas(self, filter_func) -> list[Electrolyte]:
        """
        根据自定义过滤函数筛选配方。

        Args:
            filter_func: 接受 Electrolyte 对象并返回 bool 的函数
        """
        return [f for f in self.formulas if filter_func(f)]

    def get_performance_statistics(self, metric: str = "conductivity") -> dict:
        """
        获取指定性能指标的详细统计信息。
        """
        values = [
            f.performance[metric]
            for f in self.formulas
            if metric in f.performance and f.performance[metric] is not None
        ]

        if not values:
            return {"count": 0}

        # 转换为numpy数组进行统计计算
        values_array = np.array(values)

        # 基础统计
        basic_stats = {
            "count": len(values),
            "min": float(values_array.min()),
            "max": float(values_array.max()),
            "mean": float(values_array.mean()),
            "median": float(np.median(values_array)),
            "std": float(values_array.std()),
            "var": float(values_array.var()),
        }

        # 分位数统计
        percentiles = {
            "q25": float(np.percentile(values_array, 25)),
            "q75": float(np.percentile(values_array, 75)),
            "q95": float(np.percentile(values_array, 95)),
            "q99": float(np.percentile(values_array, 99)),
        }

        # 分布特征
        distribution_stats = {
            "skewness": float(stats.skew(values_array)),
            "kurtosis": float(stats.kurtosis(values_array)),
            "iqr": percentiles["q75"] - percentiles["q25"],
        }

        # 异常值检测 (IQR方法)
        iqr = distribution_stats["iqr"]
        lower_bound = percentiles["q25"] - 1.5 * iqr
        upper_bound = percentiles["q75"] + 1.5 * iqr
        outliers = values_array[
            (values_array < lower_bound) | (values_array > upper_bound)
        ]

        outlier_stats = {
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(values) * 100,
            "outlier_values": outliers.tolist(),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

        # 合并所有统计信息
        return {
            **basic_stats,
            **percentiles,
            **distribution_stats,
            **outlier_stats,
        }

    def clear(self):
        """清空所有配方数据"""
        self.formulas.clear()
        self.electrolyte_counter = itertools.count(1)

    def copy(self) -> "ElectrolyteDataset":
        """创建数据集的深拷贝"""
        new_dataset = ElectrolyteDataset()
        new_dataset.formulas = [f for f in self.formulas]  # 浅拷贝配方列表
        return new_dataset

    # ------------------------ 温度处理方法 ------------------------ #
    def _normalize_temperature(self, temperature_k: float) -> float:
        """
        将温度标准化到 [0, 1] 范围

        Args:
            temperature_k (float): 温度值，单位为开尔文(K)

        Returns:
            float: 标准化后的温度值 [0, 1]
        """
        # 限制温度在合理范围内
        temp_clamped = max(self.TEMP_MIN, min(self.TEMP_MAX, temperature_k))

        # 标准化到 [0, 1]
        normalized = (temp_clamped - self.TEMP_MIN) / (self.TEMP_MAX - self.TEMP_MIN)

        return normalized

    def get_temperature_statistics(self) -> dict:
        """
        获取数据集中温度的详细统计信息
        """
        temperatures = []
        for formula in self.formulas:
            if (
                "temperature" in formula.condition
                and formula.condition["temperature"] is not None
            ):
                temperatures.append(formula.condition["temperature"])

        if not temperatures:
            return {"count": 0}

        # 转换为numpy数组
        temp_array = np.array(temperatures)

        # 基础统计
        basic_stats = {
            "count": len(temperatures),
            "min_k": float(temp_array.min()),
            "max_k": float(temp_array.max()),
            "mean_k": float(temp_array.mean()),
            "median_k": float(np.median(temp_array)),
            "std_k": float(temp_array.std()),
            "min_c": float(temp_array.min()) - 273.15,
            "max_c": float(temp_array.max()) - 273.15,
            "mean_c": float(temp_array.mean()) - 273.15,
            "median_c": float(np.median(temp_array)) - 273.15,
            "std_c": float(temp_array.std()),
        }

        # 分位数统计
        percentiles = {
            "q25_k": float(np.percentile(temp_array, 25)),
            "q75_k": float(np.percentile(temp_array, 75)),
            "q95_k": float(np.percentile(temp_array, 95)),
            "q25_c": float(np.percentile(temp_array, 25)) - 273.15,
            "q75_c": float(np.percentile(temp_array, 75)) - 273.15,
            "q95_c": float(np.percentile(temp_array, 95)) - 273.15,
        }

        # 分布特征
        distribution_stats = {
            "skewness": float(stats.skew(temp_array)),
            "kurtosis": float(stats.kurtosis(temp_array)),
            "iqr_k": percentiles["q75_k"] - percentiles["q25_k"],
            "iqr_c": percentiles["q75_c"] - percentiles["q25_c"],
        }

        # 归一化相关信息
        normalization_stats = {
            "normalization_range": f"{self.TEMP_MIN}K - {self.TEMP_MAX}K",
            "out_of_range_count": sum(
                1 for t in temperatures if t < self.TEMP_MIN or t > self.TEMP_MAX
            ),
            "coverage_percentage": sum(
                1 for t in temperatures if self.TEMP_MIN <= t <= self.TEMP_MAX
            )
            / len(temperatures)
            * 100,
        }

        # 温度分布区间统计
        temp_ranges = {
            "very_low_count": sum(1 for t in temperatures if t < 253.15),  # < -20°C
            "low_count": sum(
                1 for t in temperatures if 253.15 <= t < 273.15
            ),  # -20°C to 0°C
            "room_count": sum(
                1 for t in temperatures if 273.15 <= t < 303.15
            ),  # 0°C to 30°C
            "elevated_count": sum(
                1 for t in temperatures if 303.15 <= t < 333.15
            ),  # 30°C to 60°C
            "high_count": sum(1 for t in temperatures if t >= 333.15),  # >= 60°C
        }

        return {
            **basic_stats,
            **percentiles,
            **distribution_stats,
            **normalization_stats,
            **temp_ranges,
        }

    def validate_temperature_range(self):
        """
        验证数据集中的温度是否在归一化范围内
        """
        temp_stats = self.get_temperature_statistics()

        if temp_stats["count"] == 0:
            print("警告：数据集中没有温度数据")
            return

        print("温度统计信息：")
        print(f"  样本数量: {temp_stats['count']}")
        print(
            f"  温度范围: {temp_stats['min_k']:.2f}K - {temp_stats['max_k']:.2f}K ({temp_stats['min_c']:.2f}°C - {temp_stats['max_c']:.2f}°C)"
        )
        print(f"  归一化范围: {self.TEMP_MIN}K - {self.TEMP_MAX}K")

        if temp_stats["out_of_range_count"] > 0:
            print(
                f"  警告：{temp_stats['out_of_range_count']} 个样本的温度超出归一化范围"
            )
        else:
            print("  ✓ 所有温度样本都在归一化范围内")

    def comprehensive_data_analysis(
        self,
        metrics: list[str] = None,
        save_plots: bool = True,
        plot_dir: str = "plots",
    ) -> dict:
        """
        对数据集进行综合的统计分析并生成可视化图表

        Args:
            metrics: 要分析的性能指标列表，默认为["conductivity"]
            save_plots: 是否保存图表到文件
            plot_dir: 图表保存目录

        Returns:
            dict: 包含所有分析结果的字典
        """
        if metrics is None:
            metrics = ["conductivity"]

        # 创建保存目录
        if save_plots:
            import os

            os.makedirs(plot_dir, exist_ok=True)

        analysis_results = {}

        # 分析每个性能指标
        for metric in metrics:
            print(f"\n{'=' * 50}")
            print(f"分析性能指标: {metric}")
            print(f"{'=' * 50}")

            # 获取统计信息
            perf_stats = self.get_performance_statistics(metric)
            analysis_results[metric] = perf_stats

            if perf_stats["count"] == 0:
                print(f"警告：没有找到 {metric} 的有效数据")
                continue

            # 打印统计摘要
            self._print_performance_summary(metric, perf_stats)

            # 生成可视化图表
            if save_plots:
                self._plot_performance_distribution(metric, plot_dir)
                self._plot_performance_boxplot(metric, plot_dir)

        # 温度分析
        print(f"\n{'=' * 50}")
        print("温度数据分析")
        print(f"{'=' * 50}")

        temp_stats = self.get_temperature_statistics()
        analysis_results["temperature"] = temp_stats

        if temp_stats["count"] > 0:
            self._print_temperature_summary(temp_stats)
            if save_plots:
                self._plot_temperature_distribution(plot_dir)
        else:
            print("警告：没有找到温度数据")

        # 相关性分析
        if len(metrics) > 1:
            print(f"\n{'=' * 50}")
            print("相关性分析")
            print(f"{'=' * 50}")
            corr_results = self._analyze_correlations(metrics)
            analysis_results["correlations"] = corr_results
            if save_plots:
                self._plot_correlation_matrix(metrics, plot_dir)

        # 数据质量报告
        print(f"\n{'=' * 50}")
        print("数据质量报告")
        print(f"{'=' * 50}")
        quality_report = self._generate_quality_report(metrics)
        analysis_results["data_quality"] = quality_report
        self._print_quality_report(quality_report)

        return analysis_results

    def _print_performance_summary(self, metric: str, perf_stats: dict):
        """打印性能指标统计摘要"""
        print(f"\n📊 {metric} 统计摘要:")
        print(f"  样本数量: {perf_stats['count']}")
        print(f"  均值: {perf_stats['mean']:.4f}")
        print(f"  中位数: {perf_stats['median']:.4f}")
        print(f"  标准差: {perf_stats['std']:.4f}")
        print(f"  最小值: {perf_stats['min']:.4f}")
        print(f"  最大值: {perf_stats['max']:.4f}")
        print(f"  四分位距: {perf_stats['iqr']:.4f}")
        print(f"  偏度: {perf_stats['skewness']:.4f}")
        print(f"  峰度: {perf_stats['kurtosis']:.4f}")
        print(
            f"  异常值: {perf_stats['outlier_count']} ({perf_stats['outlier_percentage']:.2f}%)"
        )

    def _print_temperature_summary(self, temp_stats: dict):
        """打印温度统计摘要"""
        print(f"\n🌡️ 温度统计摘要:")
        print(f"  样本数量: {temp_stats['count']}")
        print(
            f"  温度范围: {temp_stats['min_k']:.2f}K - {temp_stats['max_k']:.2f}K ({temp_stats['min_c']:.2f}°C - {temp_stats['max_c']:.2f}°C)"
        )
        print(f"  均值: {temp_stats['mean_k']:.2f}K ({temp_stats['mean_c']:.2f}°C)")
        print(
            f"  中位数: {temp_stats['median_k']:.2f}K ({temp_stats['median_c']:.2f}°C)"
        )
        print(f"  标准差: {temp_stats['std_k']:.2f}K")
        print(f"  归一化覆盖率: {temp_stats['coverage_percentage']:.1f}%")
        print(f"  温度分布:")
        print(f"    极低温(<-20°C): {temp_stats['very_low_count']}")
        print(f"    低温(-20°C~0°C): {temp_stats['low_count']}")
        print(f"    室温(0°C~30°C): {temp_stats['room_count']}")
        print(f"    中温(30°C~60°C): {temp_stats['elevated_count']}")
        print(f"    高温(≥60°C): {temp_stats['high_count']}")

    def _plot_performance_distribution(self, metric: str, plot_dir: str):
        """绘制性能指标分布图"""
        values = [
            f.performance[metric]
            for f in self.formulas
            if metric in f.performance and f.performance[metric] is not None
        ]

        if not values:
            return

        plt.figure(figsize=(12, 8))

        # 创建2x2子图
        plt.subplot(2, 2, 1)
        plt.hist(values, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        plt.title(f"{metric} 分布直方图")
        plt.xlabel(metric)
        plt.ylabel("频次")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.boxplot(values)
        plt.title(f"{metric} 箱线图")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"{metric} Q-Q图 (正态性检验)")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        sns.violinplot(y=values)
        plt.title(f"{metric} 小提琴图")
        plt.ylabel(metric)

        plt.tight_layout()
        plt.savefig(
            f"{plot_dir}/{metric}_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_performance_boxplot(self, metric: str, plot_dir: str):
        """绘制性能指标详细箱线图"""
        values = [
            f.performance[metric]
            for f in self.formulas
            if metric in f.performance and f.performance[metric] is not None
        ]

        if not values:
            return

        plt.figure(figsize=(10, 6))
        box_plot = plt.boxplot(values, patch_artist=True)
        box_plot["boxes"][0].set_facecolor("lightblue")

        # 添加统计信息文本
        stats_info = self.get_performance_statistics(metric)
        text_info = f"""统计信息:
均值: {stats_info["mean"]:.4f}
中位数: {stats_info["median"]:.4f}
标准差: {stats_info["std"]:.4f}
异常值: {stats_info["outlier_count"]}个"""

        plt.text(
            1.1,
            plt.ylim()[1] * 0.8,
            text_info,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"),
        )

        plt.title(f"{metric} 详细箱线图")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plot_dir}/{metric}_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_temperature_distribution(self, plot_dir: str):
        """绘制温度分布图"""
        temperatures = []
        for formula in self.formulas:
            if (
                "temperature" in formula.performance
                and formula.performance["temperature"] is not None
            ):
                temperatures.append(formula.performance["temperature"])

        if not temperatures:
            return

        # 转换为摄氏度
        temps_c = [t - 273.15 for t in temperatures]

        plt.figure(figsize=(15, 10))

        # 2x3布局
        plt.subplot(2, 3, 1)
        plt.hist(temps_c, bins=20, alpha=0.7, color="orange", edgecolor="black")
        plt.title("温度分布直方图")
        plt.xlabel("温度 (°C)")
        plt.ylabel("频次")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 2)
        plt.boxplot(temps_c)
        plt.title("温度箱线图")
        plt.ylabel("温度 (°C)")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 3)
        sns.violinplot(y=temps_c)
        plt.title("温度小提琴图")
        plt.ylabel("温度 (°C)")

        plt.subplot(2, 3, 4)
        # 温度范围分布饼图
        temp_stats = self.get_temperature_statistics()
        labels = [
            "极低温\n(<-20°C)",
            "低温\n(-20~0°C)",
            "室温\n(0~30°C)",
            "中温\n(30~60°C)",
            "高温\n(≥60°C)",
        ]
        sizes = [
            temp_stats["very_low_count"],
            temp_stats["low_count"],
            temp_stats["room_count"],
            temp_stats["elevated_count"],
            temp_stats["high_count"],
        ]
        colors = ["lightblue", "lightgreen", "lightyellow", "orange", "lightcoral"]

        # 过滤掉为0的部分
        filtered_data = [
            (label, size, color)
            for label, size, color in zip(labels, sizes, colors)
            if size > 0
        ]
        if filtered_data:
            labels, sizes, colors = zip(*filtered_data)
            plt.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
        plt.title("温度范围分布")

        plt.subplot(2, 3, 5)
        # 归一化温度分布
        normalized_temps = [self._normalize_temperature(t) for t in temperatures]
        plt.hist(
            normalized_temps, bins=20, alpha=0.7, color="purple", edgecolor="black"
        )
        plt.title("归一化温度分布")
        plt.xlabel("归一化温度 [0,1]")
        plt.ylabel("频次")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 6)
        # 时间序列图（如果有多个样本）
        plt.plot(range(len(temps_c)), sorted(temps_c), "o-", markersize=3)
        plt.title("温度值排序图")
        plt.xlabel("样本索引")
        plt.ylabel("温度 (°C)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{plot_dir}/temperature_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _analyze_correlations(self, metrics: list[str]) -> dict:
        """分析不同性能指标之间的相关性"""
        correlations = {}

        # 获取所有指标的数据
        data_matrix = []
        for metric in metrics:
            values = []
            for formula in self.formulas:
                if (
                    metric in formula.performance
                    and formula.performance[metric] is not None
                ):
                    values.append(formula.performance[metric])
                else:
                    values.append(np.nan)
            data_matrix.append(values)

        # 计算相关系数矩阵
        data_array = np.array(data_matrix)
        corr_matrix = np.corrcoef(data_array)

        # 存储相关性结果
        for i, metric1 in enumerate(metrics):
            correlations[metric1] = {}
            for j, metric2 in enumerate(metrics):
                if i != j:
                    corr_value = corr_matrix[i, j]
                    if not np.isnan(corr_value):
                        correlations[metric1][metric2] = float(corr_value)

        return correlations

    def _plot_correlation_matrix(self, metrics: list[str], plot_dir: str):
        """绘制相关性矩阵热图"""
        # 构建数据矩阵
        data_dict = {}
        for metric in metrics:
            data_dict[metric] = []
            for formula in self.formulas:
                if (
                    metric in formula.performance
                    and formula.performance[metric] is not None
                ):
                    data_dict[metric].append(formula.performance[metric])
                else:
                    data_dict[metric].append(np.nan)

        # 创建DataFrame
        import pandas as pd

        df = pd.DataFrame(data_dict)

        # 计算相关性矩阵
        corr_matrix = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("性能指标相关性矩阵")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_quality_report(self, metrics: list[str]) -> dict:
        """生成数据质量报告"""
        total_formulas = len(self.formulas)
        report = {
            "total_samples": total_formulas,
            "metrics_coverage": {},
            "missing_data_summary": {},
            "data_completeness": 0.0,
        }

        total_data_points = 0
        missing_data_points = 0

        for metric in metrics:
            valid_count = len(self.get_formulas_with_performance(metric))
            missing_count = total_formulas - valid_count

            report["metrics_coverage"][metric] = {
                "valid_samples": valid_count,
                "missing_samples": missing_count,
                "coverage_percentage": (valid_count / total_formulas * 100)
                if total_formulas > 0
                else 0,
            }

            total_data_points += total_formulas
            missing_data_points += missing_count

        # 包含温度数据的检查
        temp_stats = self.get_temperature_statistics()
        temp_coverage = (
            (temp_stats["count"] / total_formulas * 100) if total_formulas > 0 else 0
        )
        report["metrics_coverage"]["temperature"] = {
            "valid_samples": temp_stats["count"],
            "missing_samples": total_formulas - temp_stats["count"],
            "coverage_percentage": temp_coverage,
        }

        total_data_points += total_formulas
        missing_data_points += total_formulas - temp_stats["count"]

        # 计算整体数据完整性
        report["data_completeness"] = (
            ((total_data_points - missing_data_points) / total_data_points * 100)
            if total_data_points > 0
            else 0
        )

        return report

    def _print_quality_report(self, report: dict):
        """打印数据质量报告"""
        print(f"\n📋 数据质量报告:")
        print(f"  总样本数: {report['total_samples']}")
        print(f"  整体数据完整性: {report['data_completeness']:.1f}%")
        print(f"\n  各指标覆盖情况:")

        for metric, coverage in report["metrics_coverage"].items():
            print(f"    {metric}:")
            print(f"      有效样本: {coverage['valid_samples']}")
            print(f"      缺失样本: {coverage['missing_samples']}")
            print(f"      覆盖率: {coverage['coverage_percentage']:.1f}%")

    # ------------------------ 特征提取方法 ------------------------ #
    def _generate_feature_tensor(
        self, electrolyte: Electrolyte, feature_mode: FeatureMode = "sequence"
    ) -> torch.Tensor:
        """
        根据设定的模式，为单个电解液对象生成特征张量。
        这是所有特征工程逻辑的集合点。
        """
        if feature_mode == "sequence":
            # 调用为Transformer等序列模型设计的特征生成函数
            return self._generate_sequence_features(electrolyte)

        elif feature_mode == "weighted_average":
            # 调用为MLP, XGBoost等模型设计的特征生成函数
            return self._generate_average_features(electrolyte)

        else:
            raise ValueError(f"未知的特征模式: {feature_mode}")

    def _generate_sequence_features(self, electrolyte: Electrolyte) -> torch.Tensor:
        """
        为序列模型（如Transformer）生成特征张量。
        输出形状为 (MAX_COMPONENTS, FEATURE_DIM+1)
        分子指纹(167), 材料类型(3), 物化性质(8), 组分占比(1) = 179
        [[分子指纹(167), 材料类型(3), 物化性质(8), 组分占比(1)], ...]
        """
        # 1. 初始化一个全零的张量用于填充
        sequence_tensor = torch.zeros(
            self.MAX_COMPONENTS, self.FEATURE_DIM, dtype=torch.float32
        )

        # 添加锂盐
        for i, material in enumerate(
            electrolyte.salts + electrolyte.solvents + electrolyte.additives
        ):
            if i >= self.MAX_COMPONENTS:
                print(
                    f"警告：配方 {electrolyte.id} 的组分数量超过MAX_COMPONENTS，部分组分将被忽略。"
                )
                break

            standardized_features_tensor = torch.tensor(
                material.standardized_feature_values, dtype=torch.float32
            )  # 分子指纹(167) + 材料类型(3) + 物化性质(8)
            fraction_tensor = torch.tensor(
                [electrolyte.proportions[i] * 0.01], dtype=torch.float32
            )  # 组分占比(1)

            # 将材料特征和其占比信息拼接在一起
            # 分子指纹(167), 材料类型(3), 物化性质(8), 组分占比(1)
            final_component_vector = torch.cat(
                [
                    standardized_features_tensor,
                    fraction_tensor,
                ]
            )

            # 3d. 将最终的组分向量填入序列张量
            sequence_tensor[i] = final_component_vector

        return sequence_tensor

    def _generate_average_features(self, electrolyte: Electrolyte) -> torch.Tensor:
        """
        通过加权平均法生成一个固定长度的特征向量。
        输出形状为 (FEATURE_DIM,)
        """
        # 初始化加权平均特征向量
        weighted_features = torch.zeros(self.FEATURE_DIM - 1, dtype=torch.float32)

        # 获取温度信息并标准化
        raw_temperature = electrolyte.condition["temperature"]  # 假设单位为K
        normalized_temp = self._normalize_temperature(raw_temperature)
        temperature = torch.tensor([normalized_temp], dtype=torch.float32)

        # 获取所有组分和对应的质量占比
        all_materials = electrolyte.salts + electrolyte.solvents + electrolyte.additives

        for i, material in enumerate(all_materials):
            # 获取组分的标准化特征
            material_features = torch.tensor(
                material.standardized_feature_values, dtype=torch.float32
            )

            # 获取组分的质量占比（转换为小数）
            weight = electrolyte.proportions[i] * 0.01

            # 加权累加到总特征向量，并在最后一位补0
            padded_features = torch.cat([material_features * weight])
            weighted_features += padded_features

        # 最后将温度信息添加到特征向量中
        weighted_features_with_tempture = torch.cat([weighted_features, temperature])

        return weighted_features_with_tempture

    def _generate_target_tensor(self, electrolyte: Electrolyte) -> torch.Tensor:
        """
        为单个电解液对象生成目标张量。
        """
        target_value = electrolyte.performance.get(self.target_metric, 0.0)
        # 对目标值进行 log1p 变换：log(1 + σ)
        # 这样可以避免电导率为0时的数学错误，同时保持数值稳定性
        target_value = torch.log1p(torch.tensor(target_value, dtype=torch.float32))
        return target_value

    def _generate_all_tensor(
        self, electrolyte: Electrolyte
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为单个电解液对象生成特征、温度和目标张量。
        """
        # 2. 调用内部方法，动态生成该配方的特征张量
        feature_tensor = self._generate_feature_tensor(
            electrolyte, feature_mode=self.feature_mode
        )

        # 3. 获取目标值（例如电导率），并转换为张量
        target_tensor = self._generate_target_tensor(electrolyte)

        # 获取温度信息并标准化
        raw_temperature = electrolyte.condition["temperature"]  # 假设单位为K
        normalized_temp = self._normalize_temperature(raw_temperature)
        temperature = torch.tensor([normalized_temp], dtype=torch.float32)

        return feature_tensor, temperature, target_tensor

    @staticmethod
    def create_splits(
        dataset_file: str,
        feature_mode: FeatureMode = "sequence",
        target_metric: str = "conductivity",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        random_seed: int = 42,
    ) -> tuple["ElectrolyteDataset", "ElectrolyteDataset", "ElectrolyteDataset"]:
        """
        从一个JSON文件创建并返回训练、验证、测试三个数据集实例。

        该方法会自动筛选出包含有效性能标签的样本进行划分。

        Args:
            dataset_file (str): 包含所有配方数据的JSON文件路径。
            feature_mode (str): 为所有新创建的数据集设置的特征模式。
            target_metric (str): 为所有新创建的数据集设置的目标性能指标。
            train_frac (float): 训练集所占比例。
            val_frac (float): 验证集所占比例。测试集比例将自动计算。
            random_seed (int): 用于复现随机打乱结果的种子。

        Returns:
            tuple: 包含训练集、验证集、测试集的三个 ElectrolyteDataset 实例。
        """

        # 1. 创建一个临时数据集实例以加载和筛选数据
        temp_dataset = ElectrolyteDataset(
            dataset_file=dataset_file,
            feature_mode=feature_mode,
            target_metric=target_metric,
        )

        # 2. 筛选出所有包含有效性能标签的样本，这是我们划分的基础
        labeled_formulas = temp_dataset.get_formulas_with_performance(
            metric=target_metric
        )

        if not labeled_formulas:
            raise ValueError(
                f"数据文件 '{dataset_file}' 中没有找到任何包含性能指标 '{target_metric}' 的有效样本。"
            )

        # 3. 使用随机种子来保证每次划分的结果都一样
        random.seed(random_seed)
        random.shuffle(labeled_formulas)

        # 4. 计算切分点
        n_total = len(labeled_formulas)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)

        # 5. 切分数据列表
        train_formulas = labeled_formulas[:n_train]
        val_formulas = labeled_formulas[n_train : n_train + n_val]
        test_formulas = labeled_formulas[n_train + n_val :]

        # 6. 创建三个新的、配置正确的数据集实例
        train_dataset = ElectrolyteDataset(
            target_metric=target_metric, feature_mode=feature_mode
        )
        val_dataset = ElectrolyteDataset(
            target_metric=target_metric, feature_mode=feature_mode
        )
        test_dataset = ElectrolyteDataset(
            target_metric=target_metric, feature_mode=feature_mode
        )

        # 7. 将切分好的配方列表分别赋给新的实例
        train_dataset.formulas = train_formulas
        val_dataset.formulas = val_formulas
        test_dataset.formulas = test_formulas

        print(f"数据划分完成 (随机种子={random_seed}):")
        print(f"  - 训练集样本数: {len(train_dataset)}")
        print(f"  - 验证集样本数: {len(val_dataset)}")
        print(f"  - 测试集样本数: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset
