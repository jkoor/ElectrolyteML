import torch
from typing import List, Tuple, Optional

from .models import BaseModel
from ..dataset.electrolyte_dataset import ElectrolyteDataset
from ..battery import Electrolyte


class Predictor:
    """
    预测专家。
    使用训练好的模型对新的候选数据进行预测。
    """

    def __init__(self, model: BaseModel, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Predictor 已加载模型 '{model_path}'，将在 {self.device} 上运行。")

    def run(self, dataset: ElectrolyteDataset) -> List[Tuple[Electrolyte, float]]:
        """
        对给定数据集中的所有候选配方进行预测。

        Args:
            dataset (ElectrolyteDataset): 包含候选配方的数据集。

        Returns:
            List[Tuple[Electrolyte, float]]: (配方对象, 预测值) 的元组列表。
        """
        candidates, X_candidates = dataset.get_candidate_features()
        if not candidates:
            print("数据集中没有需要预测的候选配方。")
            return []

        X_candidates = X_candidates.to(self.device)

        with torch.no_grad():
            mask = (X_candidates.sum(dim=-1) == 0).to(self.device)
            predictions = self.model.predict(X_candidates, mask).cpu().numpy().flatten()

        return list(zip(candidates, predictions))
