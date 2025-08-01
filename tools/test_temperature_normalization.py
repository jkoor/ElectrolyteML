#!/usr/bin/env python3
"""
测试温度归一化功能，并分析实际数据集的温度分布
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elml.dataset.electrolyte_dataset import ElectrolyteDataset


def test_temperature_normalization():
    """测试温度归一化功能"""

    print("=== 温度归一化测试 ===")

    # 1. 测试基本归一化功能
    dataset = ElectrolyteDataset()

    print(f"设定的温度归一化范围: {dataset.TEMP_MIN}K - {dataset.TEMP_MAX}K")
    print(
        f"对应摄氏度范围: {dataset.TEMP_MIN - 273.15:.1f}°C - {dataset.TEMP_MAX - 273.15:.1f}°C"
    )
    print()

    # 2. 尝试加载实际数据集进行分析
    try:
        data_file = "data/calisol23/calisol23.json"
        if os.path.exists(data_file):
            print(f"=== 分析实际数据集: {data_file} ===")
            dataset_with_data = ElectrolyteDataset(dataset_file=data_file)

            # 获取温度统计信息
            temp_stats = dataset_with_data.get_temperature_statistics()

            if temp_stats["count"] > 0:
                print("数据集中的温度统计:")
                print(f"  样本数量: {temp_stats['count']}")
                print(
                    f"  温度范围: {temp_stats['min_k']:.2f}K - {temp_stats['max_k']:.2f}K"
                )
                print(
                    f"  摄氏度范围: {temp_stats['min_c']:.2f}°C - {temp_stats['max_c']:.2f}°C"
                )
                print(
                    f"  平均温度: {temp_stats['mean_k']:.2f}K ({temp_stats['mean_c']:.2f}°C)"
                )
                print(f"  超出归一化范围的样本: {temp_stats['out_of_range_count']}")

                # 验证温度范围
                print("\n验证温度范围:")
                dataset_with_data.validate_temperature_range()

                # 测试几个实际温度值的归一化
                print("\n实际温度值归一化示例:")
                sample_temps = [
                    temp_stats["min_k"],
                    temp_stats["mean_k"],
                    temp_stats["max_k"],
                ]
                for temp in sample_temps:
                    normalized = dataset_with_data._normalize_temperature(temp)
                    print(f"  {temp:.2f}K ({temp - 273.15:.2f}°C) -> {normalized:.3f}")
            else:
                print("数据集中没有找到温度信息")

        else:
            print(f"数据文件 {data_file} 不存在，跳过实际数据分析")

    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("继续进行基本功能测试...")

    print("\n=== 基本归一化功能测试 ===")

    # 测试用例
    test_temperatures = [
        (253.15, "最低值 (-20°C)"),
        (273.15, "冰点 (0°C)"),
        (298.15, "室温 (25°C)"),
        (323.15, "最高值 (50°C)"),
        (200.0, "极低温 (-73°C，超出范围)"),
        (400.0, "极高温 (127°C，超出范围)"),
    ]

    print("温度归一化结果:")
    print("原始温度(K) | 摄氏度(°C) | 归一化值 | 说明")
    print("-" * 55)

    for temp_k, description in test_temperatures:
        normalized = dataset._normalize_temperature(temp_k)
        temp_c = temp_k - 273.15
        print(f"{temp_k:11.2f} | {temp_c:9.1f} | {normalized:8.3f} | {description}")

    print()
    print("=== 验证归一化逻辑 ===")

    # 验证边界值
    min_norm = dataset._normalize_temperature(dataset.TEMP_MIN)
    max_norm = dataset._normalize_temperature(dataset.TEMP_MAX)
    mid_norm = dataset._normalize_temperature((dataset.TEMP_MIN + dataset.TEMP_MAX) / 2)

    print(f"最低温度归一化值: {min_norm:.3f} (应该是 0.000)")
    print(f"最高温度归一化值: {max_norm:.3f} (应该是 1.000)")
    print(f"中间温度归一化值: {mid_norm:.3f} (应该是 0.500)")

    # 验证
    assert abs(min_norm - 0.0) < 1e-6, "最低温度归一化值应该是0"
    assert abs(max_norm - 1.0) < 1e-6, "最高温度归一化值应该是1"
    assert abs(mid_norm - 0.5) < 1e-6, "中间温度归一化值应该是0.5"

    print("✓ 所有验证通过！")


def suggest_temperature_range():
    """根据实际数据建议合适的温度归一化范围"""
    try:
        data_file = "data/calisol23/calisol23.json"
        if os.path.exists(data_file):
            dataset = ElectrolyteDataset(dataset_file=data_file)
            temp_stats = dataset.get_temperature_statistics()

            if temp_stats["count"] > 0:
                print("\n=== 温度范围建议 ===")
                print("基于实际数据的温度范围建议:")

                # 计算建议的范围（在实际范围基础上适当扩展）
                data_min = temp_stats["min_k"]
                data_max = temp_stats["max_k"]
                data_range = data_max - data_min

                # 扩展20%的范围作为缓冲
                buffer = data_range * 0.2
                suggested_min = data_min - buffer
                suggested_max = data_max + buffer

                # 确保在合理的物理范围内
                suggested_min = max(223.15, suggested_min)  # 不低于-50°C
                suggested_max = min(373.15, suggested_max)  # 不高于100°C

                print(
                    f"  数据实际范围: {data_min:.2f}K - {data_max:.2f}K ({data_min - 273.15:.1f}°C - {data_max - 273.15:.1f}°C)"
                )
                print(
                    f"  建议归一化范围: {suggested_min:.2f}K - {suggested_max:.2f}K ({suggested_min - 273.15:.1f}°C - {suggested_max - 273.15:.1f}°C)"
                )
                print(
                    f"  当前设定范围: {dataset.TEMP_MIN:.2f}K - {dataset.TEMP_MAX:.2f}K ({dataset.TEMP_MIN - 273.15:.1f}°C - {dataset.TEMP_MAX - 273.15:.1f}°C)"
                )

                if dataset.TEMP_MIN <= data_min and dataset.TEMP_MAX >= data_max:
                    print("  ✓ 当前设定范围已经覆盖了所有数据")
                else:
                    print("  ⚠ 建议调整温度归一化范围以更好地适应数据分布")

                return suggested_min, suggested_max
    except Exception as e:
        print(f"分析数据时出错: {e}")

    return None, None


if __name__ == "__main__":
    test_temperature_normalization()
    suggest_temperature_range()
