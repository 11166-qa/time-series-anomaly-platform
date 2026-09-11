from pathlib import Path

import torch

from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)

from models.lstm import LSTMForecaster


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
    / "train.csv"
)


LOOKBACK = 96
PRED_LEN = 24
BATCH_SIZE = 64

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


def count_parameters(model):
    """
    统计可训练参数数量。
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    # ========================================================
    # 1. 创建 Dataset
    # ========================================================

    dataset = TimeSeriesDataset(
        csv_path=DATA_PATH,
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
    )

    # ========================================================
    # 2. 创建 DataLoader
    # ========================================================

    loader = create_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # ========================================================
    # 3. 获取一个 Batch
    # ========================================================

    x, y = next(iter(loader))

    print("=" * 70)
    print("Input")
    print("=" * 70)

    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")

    # ========================================================
    # 4. 创建模型
    # ========================================================

    model = LSTMForecaster(
        input_size=dataset.num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN,
        output_size=dataset.num_features,
        dropout=DROPOUT,
    )

    print("\n" + "=" * 70)
    print("Model")
    print("=" * 70)

    print(model)

    print(
        f"\nTrainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ========================================================
    # 5. Forward
    # ========================================================

    prediction = model(x)

    print("\n" + "=" * 70)
    print("Forward")
    print("=" * 70)

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
    )

    print(
        f"Target shape: "
        f"{y.shape}"
    )

    # ========================================================
    # 6. Shape检查
    # ========================================================

    assert prediction.shape == y.shape, (
        f"Prediction shape "
        f"{prediction.shape} "
        f"!= target shape "
        f"{y.shape}"
    )

    print(
        "\nShape check: PASSED"
    )

    # ========================================================
    # 7. 测试 Loss
    # ========================================================

    criterion = torch.nn.MSELoss()

    loss = criterion(
        prediction,
        y,
    )

    print(
        f"Initial MSE Loss: "
        f"{loss.item():.6f}"
    )


if __name__ == "__main__":
    main()