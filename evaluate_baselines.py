from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baselines.naive import (
    persistence_forecast,
    seasonal_naive_forecast,
)
from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)
from utils.metrics import regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ETTh1"
    / "Baselines"
)

SCALER_PATH = DATA_DIR / "scaler.joblib"


LOOKBACK = 96
PRED_LEN = 24
BATCH_SIZE = 64


def inverse_transform_3d(
    data,
    scaler,
):

    original_shape = data.shape

    data_2d = data.reshape(
        -1,
        original_shape[-1],
    )

    restored = scaler.inverse_transform(
        data_2d
    )

    return restored.reshape(
        original_shape
    )


def calculate_metrics(
    targets,
    predictions,
    feature_columns,
    model_name,
):

    rows = []

    for i, feature in enumerate(
        feature_columns
    ):

        y_true = targets[
            :, :, i
        ].reshape(-1)

        y_pred = predictions[
            :, :, i
        ].reshape(-1)

        metrics = regression_metrics(
            y_true,
            y_pred,
        )

        rows.append({
            "model": model_name,
            "variable": feature,
            **metrics,
        })

    return pd.DataFrame(rows)


def run_baseline(
    dataloader,
    method,
    pred_len,
):

    predictions = []
    targets = []

    for x, y in dataloader:

        pred = method(
            x,
            pred_len,
        )

        predictions.append(
            pred.numpy()
        )

        targets.append(
            y.numpy()
        )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    targets = np.concatenate(
        targets,
        axis=0,
    )

    return predictions, targets


def main():

    print("=" * 70)
    print("ETTh1 Naive Baselines")
    print("=" * 70)

    test_dataset = TimeSeriesDataset(
        csv_path=
            DATA_DIR / "test.csv",

        context_csv_path=
            DATA_DIR / "val.csv",

        lookback=LOOKBACK,
        pred_len=PRED_LEN,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # Persistence
    # ========================================================

    persistence_pred_scaled, targets_scaled = (
        run_baseline(
            dataloader=test_loader,
            method=persistence_forecast,
            pred_len=PRED_LEN,
        )
    )

    # ========================================================
    # Seasonal Naive
    # ========================================================

    seasonal_pred_scaled, _ = (
        run_baseline(
            dataloader=test_loader,
            method=seasonal_naive_forecast,
            pred_len=PRED_LEN,
        )
    )

    # ========================================================
    # inverse transform
    # ========================================================

    targets = inverse_transform_3d(
        targets_scaled,
        scaler,
    )

    persistence_pred = inverse_transform_3d(
        persistence_pred_scaled,
        scaler,
    )

    seasonal_pred = inverse_transform_3d(
        seasonal_pred_scaled,
        scaler,
    )

    # ========================================================
    # Metrics
    # ========================================================

    persistence_metrics = (
        calculate_metrics(
            targets,
            persistence_pred,
            test_dataset.feature_columns,
            "Persistence",
        )
    )

    seasonal_metrics = (
        calculate_metrics(
            targets,
            seasonal_pred,
            test_dataset.feature_columns,
            "SeasonalNaive",
        )
    )

    all_metrics = pd.concat(
        [
            persistence_metrics,
            seasonal_metrics,
        ],
        ignore_index=True,
    )

    # ========================================================
    # Macro
    # ========================================================

    macro_metrics = (
        all_metrics
        .groupby("model")[
            [
                "R2",
                "RMSE",
                "MAE",
                "MAPE",
            ]
        ]
        .mean()
        .reset_index()
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics.to_csv(
        RESULT_DIR
        / "variable_metrics.csv",
        index=False,
    )

    macro_metrics.to_csv(
        RESULT_DIR
        / "macro_metrics.csv",
        index=False,
    )

    print(
        "\nVariable Metrics"
    )

    print(
        all_metrics.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Macro Metrics"
    )

    print(
        "=" * 70
    )

    print(
        macro_metrics.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()