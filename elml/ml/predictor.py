import json
import torch
import numpy as np
from typing import Union, List, Dict, Optional, Tuple
from torch.utils.data import DataLoader
import warnings

from .train import ElectrolyteTrainer
from .models import ElectrolyteMLP, ElectrolyteTransformer
from ..battery import Electrolyte
from ..dataset import ElectrolyteDataset, FeatureMode
from ..material import MaterialLibrary
from .. import MLibrary

# 过滤PyTorch嵌套张量的警告
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")


class ElectrolytePredictor:
    """
    电解液性能预测器

    提供多种预测模式：
    - 单配方预测
    - 批量配方预测
    - 温度扫描预测
    """

    def __init__(
        self,
        model: Union[ElectrolyteMLP, ElectrolyteTransformer],
        model_checkpoint_path: str,
        device: Optional[str] = None,
        feature_mode: "FeatureMode" = "weighted_average",
    ):
        """
        初始化预测器

        Args:
            model: 训练好的模型实例
            model_checkpoint_path: 模型检查点文件路径
            device: 计算设备 ('cuda', 'cpu')
            feature_mode: 特征模式 ('sequence' 用于transformer, 'weighted_average' 用于mlp)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.feature_mode = feature_mode

        # 加载模型权重
        if model_checkpoint_path:
            self.load_model(model_checkpoint_path)

        # 设置为评估模式
        self.model.eval()

        # 材料库实例
        self.library: MaterialLibrary = MLibrary

        self.dataset = ElectrolyteDataset()

    def load_model(self, checkpoint_path: str):
        """加载模型检查点"""
        try:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print(f"Successfully loaded model from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")

    def print_prediction_result(self, electrolyte, predicted_conductivity):
        """打印预测结果的格式化输出"""

        print("=== 电解液预测结果 ===")
        print(f"配方ID: {electrolyte.id}")
        print(
            f"配方名称: {electrolyte.name if hasattr(electrolyte, 'name') else '未命名'}"
        )

        print("\n组成:")
        # 显示组分信息
        total_idx = 0

        if electrolyte.salts:
            print("  锂盐:")
            for i, salt in enumerate(electrolyte.salts):
                proportion = electrolyte.proportions[total_idx + i]
                print(f"    {salt.name}: {proportion:.1f}%")
            total_idx += len(electrolyte.salts)

        if electrolyte.solvents:
            print("  溶剂:")
            for i, solvent in enumerate(electrolyte.solvents):
                proportion = electrolyte.proportions[total_idx + i]
                print(f"    {solvent.name}: {proportion:.1f}%")
            total_idx += len(electrolyte.solvents)

        if electrolyte.additives:
            print("  添加剂:")
            for i, additive in enumerate(electrolyte.additives):
                proportion = electrolyte.proportions[total_idx + i]
                print(f"    {additive.name}: {proportion:.1f}%")

        # 显示条件
        temperature_k = electrolyte.performance.get("temperature", 298.15)
        print("\n测试条件:")
        print(f"  温度: {temperature_k}K ({temperature_k - 273.15:.1f}°C)")

        # 显示预测结果
        print("\n预测结果:")
        print(f"  电导率: {predicted_conductivity:.6f} S/cm")
        print(f"  电导率: {predicted_conductivity * 1000:.3f} mS/cm")

    def predict_single_formula(
        self,
        electrolyte: Electrolyte,
        temperature: float = None,
        return_original_scale: bool = True,
    ) -> float:
        """
        预测单个电解液配方的性能

        Args:
            electrolyte: 电解液配方对象
            temperature: 测试温度 (K)，默认25°C
            return_original_scale: 是否返回原始尺度的结果

        Returns:
            预测的性能值
        """

        # 检查温度参数
        if temperature is not None:
            electrolyte.performance["temperature"] = temperature

        features, temperature_tensor, _ = self.dataset._generate_all_tensor(electrolyte)

        # 添加batch维度并预测
        features = features.unsqueeze(0)
        temperature_tensor = temperature_tensor.unsqueeze(0)

        with torch.no_grad():
            features = features.to(self.device)
            temperature_tensor = temperature_tensor.to(self.device)

            if isinstance(self.model, ElectrolyteTransformer):
                # 为Transformer模型创建padding mask
                padding_mask = features.sum(dim=-1) == 0
                outputs = self.model(
                    features, temperature_tensor, src_padding_mask=padding_mask
                )
            else:
                # MLP模型暂时只用features（可能需要后续修改）
                outputs = self.model(features)
            prediction = outputs

        # 返回原始电导率值
        return torch.expm1(prediction).item()

    def predict_dataset(
        self,
        dataset: ElectrolyteDataset,
        return_original_scale: bool = True,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对整个数据集进行预测

        Args:
            dataset: 电解液数据集
            return_original_scale: 是否返回原始尺度的结果
            batch_size: 批处理大小

        Returns:
            (预测值数组, 实际值数组)
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in dataloader:
                features, temperature, targets = batch_data
                features = features.to(self.device)
                temperature = temperature.to(self.device)

                # 模型预测
                if isinstance(self.model, ElectrolyteTransformer):
                    padding_mask = features.sum(dim=-1) == 0
                    outputs = self.model(
                        features, temperature, src_padding_mask=padding_mask
                    )
                else:
                    outputs = self.model(features)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # 合并结果
        predictions = np.concatenate(all_predictions, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()

        # 如果需要，转换回原始尺度
        if return_original_scale:
            predictions = np.expm1(predictions)
            targets = np.expm1(targets)

        return predictions, targets

    def temperature_scan_prediction(
        self,
        electrolyte: Electrolyte,
        temp_range: Tuple[float, float] = (253.15, 353.15),  # -20°C to 80°C
        num_points: int = 21,
        return_original_scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在温度范围内扫描预测电解液性能

        Args:
            electrolyte: 电解液配方对象
            temp_range: 温度范围 (K) (min_temp, max_temp)
            num_points: 温度点数量
            return_original_scale: 是否返回原始尺度的结果

        Returns:
            (温度数组, 预测值数组)
        """
        min_temp, max_temp = temp_range
        temperatures = np.linspace(min_temp, max_temp, num_points)

        predictions = []
        for temp in temperatures:
            pred = self.predict_single_formula(electrolyte, temp, return_original_scale)
            predictions.append(pred)

        return temperatures, np.array(predictions)

    def save_predictions_to_json(self, predictions_data: List[Dict], output_file: str):
        """
        将预测结果保存到JSON文件

        Args:
            predictions_data: 预测结果数据
            output_file: 输出文件路径
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions_data, f, ensure_ascii=False, indent=2)

        print(f"Predictions saved to {output_file}")


def create_predictor_from_workflow(
    workflow: ElectrolyteTrainer, checkpoint_name: str = "best_model.pth"
) -> ElectrolytePredictor:
    """
    从训练工作流创建预测器的便捷函数

    Args:
        workflow: 训练好的工作流实例
        checkpoint_name: 检查点文件名

    Returns:
        预测器实例
    """
    # 确定特征模式
    if isinstance(workflow.model, ElectrolyteTransformer):
        feature_mode = "sequence"
    else:
        feature_mode = "weighted_average"

    # 构建检查点路径
    checkpoint_path = f"{workflow.log_dir}/{checkpoint_name}"

    # 创建预测器
    predictor = ElectrolytePredictor(
        model=workflow.model,
        model_checkpoint_path=checkpoint_path,
        device=str(workflow.device),
        feature_mode=feature_mode,
    )

    return predictor
