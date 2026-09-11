from pathlib import Path
import time

import pandas as pd

from trainers.lstm_trainer import train_lstm


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ETTh1"
    / "LSTM"
    / "tuning"
    / "batch_size"
)


BASE_CONFIG = {
    "lookback": 336,
    "pred_len": 24,

    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,

    "learning_rate": 2e-3,
    "weight_decay": 1e-2,

    "max_epochs": 30,
    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


BATCH_SIZES = [
    32,
    64,
    128,
]


def main():

    results = []

    for batch_size in BATCH_SIZES:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"BATCH SIZE = {batch_size}"
        )

        print(
            "=" * 70
        )

        config = BASE_CONFIG.copy()

        config[
            "batch_size"
        ] = batch_size

        result_dir = (
            RESULT_ROOT
            / f"batch_{batch_size}"
        )

        start_time = (
            time.perf_counter()
        )

        result = train_lstm(
            data_dir=DATA_DIR,
            result_dir=result_dir,
            config=config,
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        results.append({
            "batch_size":
                batch_size,

            "best_val_loss":
                result[
                    "best_val_loss"
                ],

            "best_epoch":
                result[
                    "best_epoch"
                ],

            "epochs_run":
                result[
                    "epochs_run"
                ],

            "trainable_params":
                result[
                    "trainable_params"
                ],

            "training_seconds":
                elapsed_seconds,
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "best_val_loss"
        )
        .reset_index(drop=True)
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        RESULT_ROOT
        / "batch_size_tuning_summary.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Batch Size Tuning Summary"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSummary saved to:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()