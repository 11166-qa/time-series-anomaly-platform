import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    多变量多步时序预测 LSTM。

    输入:
        x: [batch_size, lookback, input_size]

    输出:
        y_hat: [batch_size, pred_len, output_size]
    """

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        pred_len=24,
        output_size=None,
        dropout=0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.dropout = dropout

        # 默认预测全部输入变量
        if output_size is None:
            output_size = input_size

        self.output_size = output_size

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ----------------------------------------------------
        # 输出层
        #
        # hidden_size
        #     ↓
        # pred_len × output_size
        # ----------------------------------------------------

        self.fc = nn.Linear(
            hidden_size,
            pred_len * output_size,
        )

    def forward(self, x):
        """
        前向传播。

        x:
            [B, T, C]

        B = batch_size
        T = lookback
        C = input_size
        """

        # ----------------------------------------------------
        # 1. LSTM
        # ----------------------------------------------------

        lstm_out, (h_n, c_n) = self.lstm(x)

        # lstm_out:
        # [B, T, hidden_size]

        # h_n:
        # [num_layers, B, hidden_size]

        # c_n:
        # [num_layers, B, hidden_size]

        # ----------------------------------------------------
        # 2. 取最后一个时间步的输出
        # ----------------------------------------------------

        last_output = lstm_out[:, -1, :]

        # shape:
        # [B, hidden_size]

        # ----------------------------------------------------
        # 3. 全连接层
        # ----------------------------------------------------

        prediction = self.fc(last_output)

        # shape:
        # [B, pred_len * output_size]

        # ----------------------------------------------------
        # 4. reshape 成多步预测形式
        # ----------------------------------------------------

        prediction = prediction.reshape(
            x.size(0),
            self.pred_len,
            self.output_size,
        )

        # shape:
        # [B, pred_len, output_size]

        return prediction