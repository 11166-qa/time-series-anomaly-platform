from pathlib import Path

from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
)


LOOKBACK = 96
PRED_LEN = 24
BATCH_SIZE = 64


def main():

    # ========================================================
    # 1. 创建 Dataset
    # ========================================================

    train_dataset = TimeSeriesDataset(
        csv_path=DATA_DIR / "train.csv",
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
    )

    val_dataset = TimeSeriesDataset(
        csv_path=DATA_DIR / "val.csv",
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
    )

    test_dataset = TimeSeriesDataset(
        csv_path=DATA_DIR / "test.csv",
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
    )

    # ========================================================
    # 2. 输出 Dataset 信息
    # ========================================================

    print("=" * 70)
    print("Dataset Information")
    print("=" * 70)

    print(
        f"Feature columns: "
        f"{train_dataset.feature_columns}"
    )

    print(
        f"Number of features: "
        f"{train_dataset.num_features}"
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # 3. 检查单个样本
    # ========================================================

    x, y = train_dataset[0]

    print("\n" + "=" * 70)
    print("Single Sample")
    print("=" * 70)

    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")

    print(f"X dtype: {x.dtype}")
    print(f"Y dtype: {y.dtype}")

    # ========================================================
    # 4. 创建 DataLoader
    # ========================================================

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # 5. 检查一个 Batch
    # ========================================================

    x_batch, y_batch = next(
        iter(train_loader)
    )

    print("\n" + "=" * 70)
    print("Batch Information")
    print("=" * 70)

    print(
        f"X batch shape: "
        f"{x_batch.shape}"
    )

    print(
        f"Y batch shape: "
        f"{y_batch.shape}"
    )

    print(
        f"X batch dtype: "
        f"{x_batch.dtype}"
    )

    print(
        f"Y batch dtype: "
        f"{y_batch.dtype}"
    )

    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )


if __name__ == "__main__":
    main()