import json
import torch
import numpy as np
from typing import Union, List, Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
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
        num_workers: int = 0,
        feature_mode: "FeatureMode" = "weighted_average",
    ):
        """
        初始化预测器

        Args:
            model: 训练好的模型实例
            model_checkpoint_path: 模型检查点文件路径
            device: 计算设备 ('cuda', 'cpu')
            num_workers: DataLoader的工作线程数
            feature_mode: 特征模式 ('sequence' 用于transformer, 'weighted_average' 用于mlp)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
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

    def predict_single_formula(
        self,
        electrolyte: Electrolyte,
        return_original_scale: bool = True,
    ) -> float:
        """
        预测单个电解液配方的性能

        Args:
            electrolyte: 电解液配方对象
            return_original_scale: 是否返回原始尺度的结果

        Returns:
            预测的性能值
        """

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
        if return_original_scale:
            return torch.expm1(prediction).item()
        return prediction.item()

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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in dataloader:
                if len(batch_data) == 3:
                    features, temperature, targets = batch_data
                    features = features.to(self.device)
                    temperature = temperature.to(self.device)
                    targets = targets.to(self.device)

                    # 判断模型类型并调用相应的forward方法
                    if isinstance(self.model, ElectrolyteTransformer):
                        # 为Transformer模型创建padding mask
                        # 检查features中全零的行，这些是填充的位置
                        padding_mask = (
                            features.sum(dim=-1) == 0
                        )  # [batch_size, seq_len]
                        outputs = self.model(
                            features, temperature, src_padding_mask=padding_mask
                        )
                    else:
                        # 对于MLP模型，temperature需要与features拼接
                        # 注意：这里假设MLP模型也需要温度信息，可能需要修改MLP模型
                        outputs = self.model(features)
                else:
                    raise ValueError(
                        f"Expected 3 elements in batch_data, got {len(batch_data)}"
                    )

                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        predictions = torch.cat(all_predictions, dim=0).detach().numpy().flatten()
        targets = torch.cat(all_targets, dim=0).detach().numpy().flatten()
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
            # 设置电解液的测试条件
            electrolyte.condition["temperature"] = temp
            pred = self.predict_single_formula(electrolyte, return_original_scale)
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
