import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """机器学习模型基类"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """前向传播"""
        raise NotImplementedError

    def predict(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """推理方法"""
        self.eval()
        with torch.no_grad():
            return self.forward(x, mask)


class ElectrolyteTransformer(BaseModel):
    """Transformer 模型，用于电解液电导率预测"""

    def __init__(
        self,
        input_dim: int = 178,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        # mask: [batch_size, seq_len], True for padded positions
        if mask is not None:
            mask = mask.bool()
        out = self.transformer(
            x, src_key_padding_mask=mask
        )  # [batch_size, seq_len, input_dim]
        out = out[:, 0, :]  # 取第一个位置（CLS-like）
        return self.fc(out)  # [batch_size, 1]


class ElectrolyteMLP(BaseModel):
    """MLP 模型，用于电解液电导率预测"""

    def __init__(
        self,
        input_dim: int = 178,
        seq_len: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__(input_dim)
        self.flatten_dim = input_dim * seq_len
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        x = x.view(x.size(0), -1)  # [batch_size, seq_len * input_dim]
        return self.net(x)  # [batch_size, 1]
