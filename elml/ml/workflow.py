"""
统一的机器学习工作流类
整合训练、预测和分析功能，确保模型和参数的一致性
"""

import os
import json
import torch
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
from torch.utils.data import Dataset
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

from .train import ElectrolyteTrainer
from .predictor import ElectrolytePredictor
from .analysis import ElectrolyteAnalyzer
from .models import ElectrolyteMLP, ElectrolyteTransformer
from ..dataset import ElectrolyteDataset, FeatureMode

# 过滤PyTorch嵌套张量的警告
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")


@dataclass
class WorkflowConfig:
    """工作流配置类"""

    # 模型配置
    model_type: str = "transformer"  # "mlp" or "transformer"
    input_dim: int = 179
    output_dim: int = 1

    # MLP特定配置
    hidden_dims: Optional[List[int]] = None
    dropout_rate: float = 0.2

    # Transformer特定配置
    model_dim: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512

    # 训练配置
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer_name: str = "adamw"
    num_epochs: int = 100
    early_stopping_patience: int = 20

    # 数据配置
    feature_mode: FeatureMode = (
        "sequence"  # "sequence" for transformer, "weighted_average" for mlp
    )

    # 路径配置
    data_path: str = ""
    log_dir: str = "runs"
    model_name: str = "model"

    # 设备配置
    device: Optional[str] = None
    num_workers: int = 0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]

        # 根据模型类型设置默认特征模式
        if self.model_type == "transformer" and self.feature_mode == "weighted_average":
            self.feature_mode = "sequence"
        elif self.model_type == "mlp" and self.feature_mode == "sequence":
            self.feature_mode = "weighted_average"


class ElectrolyteWorkflow:
    """
    电解液机器学习统一工作流类

    整合训练、预测和分析功能，确保模型和参数的一致性。
    提供完整的机器学习生命周期管理。
    """

    def __init__(self, config: Union[WorkflowConfig, Dict[str, Any], str]):
        """
        初始化工作流

        Args:
            config: 配置对象、配置字典或配置文件路径
        """
        if isinstance(config, str):
            # 从文件加载配置
            self.config = self._load_config_from_file(config)
        elif isinstance(config, dict):
            # 从字典创建配置
            self.config = WorkflowConfig(**config)
        else:
            # 直接使用配置对象
            self.config = config

        # 设置设备
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化状态
        self.model = None
        self.trainer = None
        self.predictor = None
        self.analyzer = None
        self.datasets = {}
        self.is_trained = False
        self.model_path = None

        # 创建日志目录
        self.log_dir = Path(self.config.log_dir) / self.config.model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        print(f"Workflow initialized with device: {self.device}")
        print(f"Model type: {self.config.model_type}")
        print(f"Feature mode: {self.config.feature_mode}")
        print(f"Log directory: {self.log_dir}")

    def _load_config_from_file(self, config_path: str) -> WorkflowConfig:
        """从文件加载配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return WorkflowConfig(**config_dict)

    def _save_config(self):
        """保存配置到文件"""
        config_path = self.log_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to: {config_path}")

    def setup_data(
        self, data_path: Optional[str] = None
    ) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
        """
        设置数据集

        Args:
            data_path: 数据文件路径，如果为None则使用配置中的路径

        Returns:
            训练集、验证集、测试集元组
        """
        if data_path:
            self.config.data_path = data_path

        if not self.config.data_path:
            raise ValueError("Data path not specified in config or arguments")

        print(f"Loading data from: {self.config.data_path}")
        print(f"Feature mode: {self.config.feature_mode}")

        # 创建数据集
        train_ds, val_ds, test_ds = ElectrolyteDataset.create_splits(
            self.config.data_path, feature_mode=self.config.feature_mode
        )

        self.datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

        print("Data loaded successfully:")
        print(f"  Training samples: {len(train_ds)}")
        print(f"  Validation samples: {len(val_ds)}")
        if test_ds:
            print(f"  Test samples: {len(test_ds)}")

        return train_ds, val_ds, test_ds

    def setup_model(self) -> Union[ElectrolyteMLP, ElectrolyteTransformer]:
        """
        设置模型

        Returns:
            初始化的模型实例
        """
        if self.config.model_type == "mlp":
            self.model = ElectrolyteMLP(
                input_dim=self.config.input_dim, output_dim=self.config.output_dim
            )
        elif self.config.model_type == "transformer":
            self.model = ElectrolyteTransformer(
                input_dim=self.config.input_dim,
                model_dim=self.config.model_dim,
                nhead=self.config.nhead,
                num_encoder_layers=self.config.num_encoder_layers,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout_rate,
                output_dim=self.config.output_dim,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        self.model.to(self.device)

        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Model created: {self.config.model_type}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return self.model

    def setup_trainer(self) -> ElectrolyteTrainer:
        """
        设置训练器

        Returns:
            配置好的训练器实例
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")

        if not self.datasets:
            raise ValueError("Datasets not loaded. Call setup_data() first.")

        self.trainer = ElectrolyteTrainer(
            model=self.model,
            train_dataset=self.datasets["train"],
            val_dataset=self.datasets["val"],
            test_dataset=self.datasets.get("test"),
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            device=str(self.device),
            num_workers=self.config.num_workers,
            log_dir=str(self.log_dir),
            model_checkpoint_path=str(self.log_dir / "best_model.pth"),
            optimizer_name=self.config.optimizer_name,
        )

        print("Trainer initialized successfully")
        return self.trainer

    def train(self) -> Dict[str, float]:
        """
        训练模型

        Returns:
            训练结果字典
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer() first.")

        print("=" * 50)
        print("Starting model training...")
        print("=" * 50)

        # 训练模型
        results = self.trainer.train(
            num_epochs=self.config.num_epochs,
            early_stopping_patience=self.config.early_stopping_patience,
        )

        self.is_trained = True
        self.model_path = self.log_dir / "best_model.pth"

        # 绘制训练曲线
        try:
            self.trainer.plot_training_history()
            print(f"Training history plot saved to: {self.log_dir}")
        except Exception as e:
            print(f"Warning: Could not save training plot: {e}")

        # 保存训练结果
        results_path = self.log_dir / "training_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("=" * 50)
        print("Training completed successfully!")
        print(f"Best model saved to: {self.model_path}")
        print(f"Training results saved to: {results_path}")
        print("=" * 50)

        return results

    def setup_predictor(self, model_path: Optional[str] = None) -> ElectrolytePredictor:
        """
        设置预测器

        Args:
            model_path: 模型文件路径，如果为None则使用默认路径

        Returns:
            配置好的预测器实例
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")

        if model_path is None:
            if self.model_path is None:
                model_path = str(self.log_dir / "best_model.pth")
            else:
                model_path = str(self.model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 确保feature_mode是正确的类型
        feature_mode: FeatureMode = self.config.feature_mode

        self.predictor = ElectrolytePredictor(
            model=self.model,
            model_checkpoint_path=model_path,
            device=str(self.device),
            num_workers=self.config.num_workers,
            feature_mode=feature_mode,
        )

        print(f"Predictor initialized with model: {model_path}")
        return self.predictor

    def predict(
        self, dataset: Optional["ElectrolyteDataset"] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        进行预测

        Args:
            dataset: 要预测的数据集，如果为None则使用测试集

        Returns:
            预测值和真实值的元组
        """
        if self.predictor is None:
            raise ValueError("Predictor not initialized. Call setup_predictor() first.")

        if dataset is None:
            if "test" not in self.datasets or self.datasets["test"] is None:
                raise ValueError("No test dataset available and no dataset provided")
            dataset = self.datasets["test"]

        print("Making predictions...")
        predictions, actual_values = self.predictor.predict_dataset(
            dataset, return_original_scale=True
        )

        print(f"Predictions completed for {len(predictions)} samples")
        return predictions, actual_values

    def analyze(
        self,
        dataset: Optional["ElectrolyteDataset"] = None,
        show_plots: bool = False,
        save_plots: bool = True,
        plot_filename: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        进行预测结果分析

        Args:
            dataset: 要分析的数据集，如果为None则使用测试集
            show_plots: 是否显示图表
            save_plots: 是否保存图表
            plot_filename: 图表文件名

        Returns:
            分析结果字典，如果分析失败则返回None
        """
        if self.predictor is None:
            raise ValueError("Predictor not initialized. Call setup_predictor() first.")

        if dataset is None:
            if "test" not in self.datasets or self.datasets["test"] is None:
                raise ValueError("No test dataset available and no dataset provided")
            dataset = self.datasets["test"]

        # 初始化分析器（如果还没有初始化）
        if self.analyzer is None:
            analysis_dir = str(self.log_dir / "analysis")
            self.analyzer = ElectrolyteAnalyzer(
                predictor=self.predictor, save_dir=analysis_dir
            )

        if plot_filename is None:
            plot_filename = str(self.log_dir / "prediction_analysis.png")
        elif not os.path.isabs(plot_filename):
            plot_filename = str(self.log_dir / plot_filename)

        print("=" * 50)
        print("Starting prediction analysis...")
        print("=" * 50)

        # 进行分析
        results = self.analyzer.analyze_dataset(
            dataset=dataset,
            show_plots=show_plots,
            save_plots=save_plots,
            plot_filename=plot_filename,
        )

        if results is None:
            print("Analysis returned no results")
            return None

        # 保存分析结果
        analysis_results_path = self.log_dir / "analysis_results.json"
        self.analyzer.save_results(str(analysis_results_path))

        print("=" * 50)
        print("Analysis completed successfully!")
        print(f"Analysis results saved to: {analysis_results_path}")
        if save_plots:
            print(f"Analysis plots saved to: {plot_filename}")
        print("=" * 50)

        return results

    def run_full_pipeline(
        self,
        data_path: Optional[str] = None,
        skip_training: bool = False,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的机器学习流水线

        Args:
            data_path: 数据文件路径
            skip_training: 是否跳过训练（使用已有模型）
            model_path: 如果跳过训练，指定要加载的模型路径

        Returns:
            包含所有结果的字典
        """
        results = {}

        try:
            # 1. 设置数据
            self.setup_data(data_path)

            # 2. 设置模型
            self.setup_model()

            # 3. 训练或加载模型
            if not skip_training:
                # 设置训练器并训练
                self.setup_trainer()
                training_results = self.train()
                results["training"] = training_results
            else:
                if model_path:
                    self.model_path = Path(model_path)
                print("Skipping training, using existing model")

            # 4. 设置预测器
            self.setup_predictor(model_path)

            # 5. 进行预测和分析
            analysis_results = self.analyze()
            results["analysis"] = analysis_results

            # 6. 评估性能（如果有测试集）
            if self.datasets.get("test") and self.trainer:
                test_loss = self.trainer.evaluate(self.trainer.test_loader)
                results["test_loss"] = test_loss
                print(f"Final test loss: {test_loss:.6f}")

            results["status"] = "success"
            results["model_path"] = str(self.model_path)
            results["log_dir"] = str(self.log_dir)

        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        # 保存完整结果
        full_results_path = self.log_dir / "full_results.json"
        with open(full_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print("=" * 50)
        print("Full pipeline completed successfully!")
        print(f"Complete results saved to: {full_results_path}")
        print("=" * 50)

        return results

    def load_from_checkpoint(self, log_dir: str):
        """
        从检查点加载工作流状态

        Args:
            log_dir: 日志目录路径
        """
        log_path = Path(log_dir)

        # 加载配置
        config_path = log_path / "config.json"
        if config_path.exists():
            self.config = self._load_config_from_file(str(config_path))
            self.log_dir = log_path

            # 重新设置模型
            self.setup_model()

            # 检查是否有最佳模型
            model_path = log_path / "best_model.pth"
            if model_path.exists():
                self.model_path = model_path
                self.is_trained = True
                print(f"Loaded workflow from checkpoint: {log_dir}")
            else:
                print(f"Loaded configuration but no trained model found in: {log_dir}")
        else:
            raise FileNotFoundError(f"No configuration found in: {config_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        if self.model is None:
            return {"status": "Model not initialized"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        info = {
            "model_type": self.config.model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "is_trained": self.is_trained,
            "model_path": str(self.model_path) if self.model_path else None,
            "config": asdict(self.config),
        }

        return info

    def get_analyzer(self) -> Optional[ElectrolyteAnalyzer]:
        """
        获取分析器实例

        Returns:
            分析器实例，如果预测器未初始化则返回None
        """
        if self.predictor is None:
            print("Warning: Predictor not initialized. Call setup_predictor() first.")
            return None

        if self.analyzer is None:
            analysis_dir = str(self.log_dir / "analysis")
            self.analyzer = ElectrolyteAnalyzer(
                predictor=self.predictor, save_dir=analysis_dir
            )

        return self.analyzer

    def compare_with_other_model(
        self,
        other_workflow: "ElectrolyteWorkflow",
        dataset: Optional["ElectrolyteDataset"] = None,
        model_names: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        与另一个工作流的模型进行比较

        Args:
            other_workflow: 另一个工作流实例
            dataset: 测试数据集，如果为None则使用当前工作流的测试集
            model_names: 模型名称元组，默认使用模型类型

        Returns:
            模型比较结果
        """
        if self.predictor is None:
            raise ValueError("Current workflow predictor not initialized")
        if other_workflow.predictor is None:
            raise ValueError("Other workflow predictor not initialized")

        if dataset is None:
            if "test" not in self.datasets or self.datasets["test"] is None:
                raise ValueError("No test dataset available")
            dataset = self.datasets["test"]

        if model_names is None:
            model_names = (
                f"{self.config.model_type}_{self.config.model_name}",
                f"{other_workflow.config.model_type}_{other_workflow.config.model_name}",
            )

        # 获取分析器
        analyzer1 = self.get_analyzer()
        analyzer2 = other_workflow.get_analyzer()

        if analyzer1 is None or analyzer2 is None:
            raise ValueError("Failed to initialize analyzers")

        # 进行模型比较
        comparison_results = analyzer1.compare_models(
            other_analyzer=analyzer2, dataset=dataset, model_names=model_names
        )

        # 保存比较结果
        comparison_path = self.log_dir / "model_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)

        print(f"Model comparison results saved to: {comparison_path}")
        return comparison_results

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        获取分析结果摘要

        Returns:
            分析结果摘要，如果未进行分析则返回空字典
        """
        if self.analyzer is None:
            return {"error": "No analyzer available. Run analyze() first."}

        return self.analyzer.get_summary()

    def __repr__(self) -> str:
        return f"ElectrolyteWorkflow(model_type={self.config.model_type}, is_trained={self.is_trained})"
