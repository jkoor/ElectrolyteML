import torch
import torch.nn as nn
from typing import Optional


class BaseModel(nn.Module):
    """
    所有机器学习模型的基类。
    它定义了一个标准的接口，以便与 Trainer 和 Predictor 类兼容。
    """

    def __init__(self, input_dim: int):
        """
        初始化基类。

        Args:
            input_dim (int): 单个输入项（例如一个材料）的特征维度。
        """
        super().__init__()
        self.input_dim = input_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        定义模型的前向传播逻辑。
        这是训练的核心，必须由所有子类实现。

        Args:
            x (torch.Tensor): 输入的特征张量，形状为 [batch_size, seq_len, input_dim]。
            mask (torch.Tensor, optional): 一个布尔或浮点型的掩码张量，
                                           用于指示哪些输入是填充项，形状为 [batch_size, seq_len]。
                                           在Transformer中尤其重要。

        Returns:
            torch.Tensor: 模型的输出，通常形状为 [batch_size, 1]。
        """
        raise NotImplementedError("每个子类都必须实现自己的 forward 方法！")

    def predict(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        定义模型的推理（预测）逻辑。
        这个方法会以无梯度计算的模式调用 forward 方法，以提高效率。
        """
        self.eval()  # 切换到评估模式 (例如关闭 Dropout)
        with torch.no_grad():
            return self.forward(x, mask)


class ElectrolyteTransformer(BaseModel):
    """
    一个基于 Transformer Encoder 的模型，用于预测电解液电导率。
    它能有效捕捉配方中不同材料之间的相互作用。
    """

    def __init__(
        self,
        input_dim: int = 178,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        """
        初始化 Transformer 模型。

        Args:
            input_dim (int): 输入特征维度。
            hidden_dim (int): Transformer 模型的核心维度 (d_model)。
            n_layers (int): Transformer Encoder 的层数。
            n_heads (int): 多头注意力机制中的头数。
            dropout (float): Dropout 的比率。
        """
        super().__init__(input_dim)
        
        # --- 关键修改 ---
        # 1. 添加一个嵌入层，将 input_dim 映射到 hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 2. 定义 Transformer Encoder 层，使用 hidden_dim 作为 d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,  # 通常前馈网络维度是 d_model 的 4 倍
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        # 3. 定义最终的线性输出层，输入维度为 hidden_dim
        self.fc_out = nn.Linear(hidden_dim, 1)
        # --- 修改结束 ---

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transformer 模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状 [batch_size, seq_len, input_dim]。
            mask (torch.Tensor): 填充掩码，形状 [batch_size, seq_len]。
                                 值为 True 的位置代表是填充项，需要被忽略。

        Returns:
            torch.Tensor: 预测的电导率，形状 [batch_size, 1]。
        """
        # --- 关键修改 ---
        # 1. 首先通过嵌入层
        x = self.embedding(x)
        
        # 2. Transformer Encoder 处理
        out = self.transformer_encoder(src=x, src_key_padding_mask=mask)
        # --- 修改结束 ---

        # 我们使用序列的第一个位置（类似[CLS] token）的输出来代表整个序列的特征
        out = out[:, 0, :]

        # 通过线性层得到最终输出
        return self.fc_out(out)


class ElectrolyteMLP(BaseModel):
    """
    一个基于多层感知机 (MLP) 的模型。
    它将所有材料的特征展平后，通过全连接网络进行预测。
    """

    def __init__(
        self,
        input_dim: int = 178,
        seq_len: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        """
        初始化 MLP 模型。

        Args:
            input_dim (int): 单个材料的特征维度。
            seq_len (int): 序列长度（即配方中材料的最大数量）。
            hidden_dim (int): 隐藏层的维度。
            dropout (float): Dropout 的比率。
        """
        super().__init__(input_dim)
        self.flatten_dim = input_dim * seq_len

        # 定义一个序列化的全连接网络
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        MLP 模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状 [batch_size, seq_len, input_dim]。
            mask (torch.Tensor, optional): 此模型中未使用掩码，但保留以保证接口统一。

        Returns:
            torch.Tensor: 预测的电导率，形状 [batch_size, 1]。
        """
        # 将 [batch_size, seq_len, input_dim] 展平为 [batch_size, seq_len * input_dim]
        x = x.view(x.size(0), -1)
        return self.net(x)
