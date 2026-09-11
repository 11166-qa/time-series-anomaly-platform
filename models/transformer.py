import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    输入/输出：
        [B, T, d_model]
    """

    def __init__(
        self,
        d_model,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(
            dropout
        )

        # [max_len, d_model]
        pe = torch.zeros(
            max_len,
            d_model,
        )

        # [max_len, 1]
        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # buffer:
        # 属于模型状态，但不是可训练参数
        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        # x:
        # [B, T, d_model]

        x = (
            x
            + self.pe[
                :,
                :x.size(1),
                :
            ]
        )

        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    基于 Transformer Encoder 的
    多变量多步时序预测模型。

    Input:
        [B, lookback, input_size]

    Output:
        [B, pred_len, output_size]
    """

    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        pred_len=24,
        output_size=None,
        dropout=0.1,
    ):
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = (
            dim_feedforward
        )
        self.pred_len = pred_len
        self.output_size = output_size
        self.dropout = dropout

        # ====================================================
        # 1. Input Projection
        # ====================================================

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        # ====================================================
        # 2. Positional Encoding
        # ====================================================

        self.positional_encoding = (
            PositionalEncoding(
                d_model=d_model,
                dropout=dropout,
            )
        )

        # ====================================================
        # 3. Transformer Encoder
        # ====================================================

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=
                    dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer=
                    encoder_layer,

                num_layers=
                    num_layers,
            )
        )

        # ====================================================
        # 4. Prediction Head
        # ====================================================

        self.output_norm = (
            nn.LayerNorm(
                d_model
            )
        )

        self.fc = nn.Linear(
            d_model,
            pred_len * output_size,
        )

    def forward(self, x):

        # ----------------------------------------------------
        # x
        # [B, T, input_size]
        # ----------------------------------------------------

        x = self.input_projection(x)

        # [B, T, d_model]

        # Transformer 中通常会进行尺度调整
        x = x * math.sqrt(
            self.d_model
        )

        x = (
            self.positional_encoding(
                x
            )
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        encoded = self.encoder(x)

        # [B, T, d_model]

        # ----------------------------------------------------
        # 使用最后一个时间位置的表示
        # ----------------------------------------------------

        representation = (
            encoded[:, -1, :]
        )

        # [B, d_model]

        representation = (
            self.output_norm(
                representation
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = self.fc(
            representation
        )

        # [B, pred_len * output_size]

        prediction = (
            prediction.reshape(
                x.size(0),
                self.pred_len,
                self.output_size,
            )
        )

        return prediction