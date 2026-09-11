from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    多变量时序预测数据集。

    输入:
        过去 lookback 个时间点

    输出:
        未来 pred_len 个时间点
    """

    def __init__(
            self,
            csv_path,
            lookback=96,
            pred_len=24,
            timestamp_column="date",
            context_csv_path=None,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.lookback = lookback
        self.pred_len = pred_len
        self.timestamp_column = timestamp_column

        # ====================================================
        # 1. 检查文件
        # ====================================================

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"找不到数据文件：{self.csv_path}"
            )

        # ====================================================
        # 2. 读取 CSV
        # ====================================================

        # 当前数据集
        df = pd.read_csv(self.csv_path)

        # 如果提供前序数据，则取前序数据最后 lookback 个时间点
        # 作为当前数据集最前面的历史上下文
        if context_csv_path is not None:

            context_csv_path = Path(context_csv_path)

            if not context_csv_path.exists():
                raise FileNotFoundError(
                    f"找不到上下文数据文件：{context_csv_path}"
                )

            context_df = pd.read_csv(
                context_csv_path
            )

            # 只需要最后 lookback 个时间点
            context_df = (
                context_df
                .tail(self.lookback)
                .copy()
            )

            if len(context_df) < self.lookback:
                raise ValueError(
                    f"上下文数据不足，需要至少 "
                    f"{self.lookback} 个时间点。"
                )

            # 拼接：
            # 前一个数据集最后 lookback 行 + 当前数据集
            df = pd.concat(
                [context_df, df],
                axis=0,
                ignore_index=True,
            )

        # ====================================================
        # 3. 检查时间列
        # ====================================================

        if self.timestamp_column not in df.columns:
            raise ValueError(
                f"数据中不存在时间列："
                f"{self.timestamp_column}"
            )

        df[self.timestamp_column] = pd.to_datetime(
            df[self.timestamp_column],
            errors="raise",
        )

        # 保存时间戳，后面预测结果和异常检测会使用
        self.timestamps = (
            df[self.timestamp_column]
            .reset_index(drop=True)
        )

        # ====================================================
        # 4. 获取数值变量
        # ====================================================

        self.feature_columns = [
            col
            for col in df.columns
            if col != self.timestamp_column
        ]

        if len(self.feature_columns) == 0:
            raise ValueError(
                "没有找到可以用于建模的数值变量。"
            )

        # ====================================================
        # 5. 转换成 NumPy float32
        # ====================================================

        values = (
            df[self.feature_columns]
            .to_numpy(dtype=np.float32)
        )

        # shape:
        # [time_steps, features]

        if np.isnan(values).any():
            raise ValueError(
                "输入数据中存在 NaN。"
            )

        if np.isinf(values).any():
            raise ValueError(
                "输入数据中存在 Inf。"
            )

        self.data = values

        # ====================================================
        # 6. 检查数据长度
        # ====================================================

        minimum_length = (
            self.lookback
            + self.pred_len
        )

        if len(self.data) < minimum_length:
            raise ValueError(
                f"数据长度不足。"
                f"至少需要 {minimum_length} 条数据，"
                f"实际只有 {len(self.data)} 条。"
            )

    def __len__(self):
        """
        返回能够构造出的滑动窗口样本数量。
        """

        return (
            len(self.data)
            - self.lookback
            - self.pred_len
            + 1
        )

    def __getitem__(self, index):
        """
        根据 index 构造一个时序预测样本。
        """

        # 输入窗口
        x_start = index
        x_end = index + self.lookback

        # 预测窗口
        y_start = x_end
        y_end = y_start + self.pred_len

        x = self.data[
            x_start:x_end
        ]

        y = self.data[
            y_start:y_end
        ]

        # NumPy -> PyTorch Tensor
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y

    @property
    def num_features(self):
        """
        返回变量数量。
        """

        return len(self.feature_columns)

    def get_target_timestamps(self, index):
        """
        获取某一个样本对应预测区间的时间戳。

        后续预测结果保存和异常检测时使用。
        """

        start = index + self.lookback
        end = start + self.pred_len

        return self.timestamps.iloc[
            start:end
        ].to_numpy()


def create_dataloader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
):
    """
    创建 PyTorch DataLoader。
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )