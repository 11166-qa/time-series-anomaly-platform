from pathlib import Path

from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)

from models.transformer import (
    TransformerForecaster,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
    / "train.csv"
)


LOOKBACK = 336
PRED_LEN = 24
BATCH_SIZE = 64


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    dataset = TimeSeriesDataset(
        csv_path=DATA_PATH,
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
    )

    loader = create_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    x, y = next(iter(loader))

    model = TransformerForecaster(
        input_size=
            dataset.num_features,

        d_model=64,

        nhead=4,

        num_layers=2,

        dim_feedforward=128,

        pred_len=PRED_LEN,

        output_size=
            dataset.num_features,

        dropout=0.1,
    )

    prediction = model(x)

    print("=" * 70)

    print(
        f"Input shape:      "
        f"{x.shape}"
    )

    print(
        f"Target shape:     "
        f"{y.shape}"
    )

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
    )

    print(
        f"Parameters:       "
        f"{count_parameters(model):,}"
    )

    assert (
        prediction.shape
        == y.shape
    )

    print(
        "\nTransformer shape check: PASSED"
    )


if __name__ == "__main__":
    main()