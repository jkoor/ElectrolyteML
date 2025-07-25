import math
import torch
import torch.nn as nn


class ElectrolyteMLP(nn.Module):
    """
    一个简单的多层感知机（MLP），用于处理加权平均后的电解液特征。

    输入是一个固定长度的向量，代表整个电解液的平均特性。
    """

    def __init__(self, input_dim: int = 178, output_dim: int = 1):
        """
        初始化MLP模型。

        Args:
            input_dim (int): 输入特征的维度。
                             根据 `ElectrolyteDataset`，该值为 3+167+8 = 178。
            output_dim (int): 输出的维度，通常为1（预测的目标值）。
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        Returns:
            torch.Tensor: 模型的输出，形状为 (batch_size, output_dim)。
        """
        return self.model(x)


class PositionalEncoding(nn.Module):
    """
    为Transformer模型注入位置信息。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 修复：创建适合batch_first=True的位置编码
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为buffer，形状为 [max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # pe[:seq_len] 形状为 [seq_len, d_model]
        # 广播到 [batch_size, seq_len, d_model]
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class ElectrolyteTransformer(nn.Module):
    """
    一个基于Transformer的序列模型，用于处理电解液的组分序列特征。

    输入是一个变长的序列，每个元素代表一种材料及其占比。
    """

    def __init__(
        self,
        input_dim: int = 179,  # 3 (材料类型) + 166 (分子指纹) + 8 (物化性质) + 1 (组分占比)
        model_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        """
        初始化Transformer模型。

        Args:
            input_dim (int): 输入序列中每个元素的特征维度。
                             根据 `ElectrolyteDataset`，该值为 178 (特征) + 1 (占比) = 179。
            model_dim (int): Transformer模型内部的特征维度 (d_model)。
            nhead (int): 多头注意力机制中的头数。
            num_encoder_layers (int): Transformer编码器的层数。
            dim_feedforward (int): 前馈网络（FFN）的维度。
            dropout (float): Dropout的比率。
            output_dim (int): 输出的维度，通常为1。
        """
        super().__init__()
        self.model_dim = model_dim

        # 1. 输入层：将输入特征维度映射到模型内部维度
        self.input_embedding = nn.Linear(input_dim, model_dim)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 使用 batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 4. 输出层：将Transformer的输出映射到最终预测值
        self.output_layer = nn.Linear(model_dim, output_dim)

        # 温度编码器
        self.temperature_encoder = nn.Sequential(
            nn.Linear(1, model_dim // 4),
            nn.ReLU(),
            nn.Linear(model_dim // 4, model_dim),
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, src: torch.Tensor, temperature: torch.Tensor, src_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Transformer的前向传播。

        Args:
            src (torch.Tensor): 输入序列张量，形状为 (batch_size, seq_len, input_dim)。
            temperature: [batch_size, 1] 温度值
            src_padding_mask (torch.Tensor, optional): 用于屏蔽填充部分的掩码。
                                                       形状为 (batch_size, seq_len)。

        Returns:
            torch.Tensor: 模型的输出，形状为 (batch_size, output_dim)。
        """
        # 1. 输入嵌入
        embedded = self.input_embedding(src) * math.sqrt(self.model_dim)

        # 2. 添加位置编码 - 修复后可以正常使用
        embedded = self.pos_encoder(embedded)

        # 3. Transformer编码器
        transformer_output = self.transformer_encoder(
            embedded, src_key_padding_mask=src_padding_mask
        )

        # 4. 序列聚合 (现有逻辑很好)
        if src_padding_mask is not None:
            mask = ~src_padding_mask.unsqueeze(-1).bool()
            masked_output = transformer_output * mask
            valid_lengths = mask.sum(dim=1)
            aggregated_output = masked_output.sum(dim=1) / valid_lengths
        else:
            aggregated_output = transformer_output.mean(dim=1)

        # 5. 温度编码
        temperature_encoded = self.temperature_encoder(temperature)  # [batch_size, model_dim]

        # 6. 融合配方特征和温度特征
        fused_features = torch.cat(
            [aggregated_output, temperature_encoded], dim=-1
        )  # [batch_size, model_dim*2]
        fused_output = self.fusion_layer(fused_features)  # [batch_size, model_dim]

        # 7. 输出预测
        output = self.output_layer(fused_output)
        return output
