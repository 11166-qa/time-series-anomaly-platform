from pathlib import Path

import pandas as pd

from trainers.lstm_trainer import (
    train_lstm,
)


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
    / "lookback"
)


BASE_CONFIG = {

    "pred_len": 24,

    "batch_size": 64,

    "hidden_size": 64,

    "num_layers": 2,

    "dropout": 0.2,

    "learning_rate": 1e-3,

    "weight_decay": 1e-4,

    "max_epochs": 30,

    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


LOOKBACK_VALUES = [
    48,
    96,
    168,
    336,
]


def main():

    results = []

    for lookback in LOOKBACK_VALUES:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"LOOKBACK = {lookback}"
        )

        print(
            "=" * 70
        )

        config = (
            BASE_CONFIG.copy()
        )

        config[
            "lookback"
        ] = lookback

        result_dir = (
            RESULT_ROOT
            / f"lookback_{lookback}"
        )

        result = train_lstm(
            data_dir=
                DATA_DIR,

            result_dir=
                result_dir,

            config=
                config,
        )

        results.append({
            "lookback":
                lookback,

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
        })

    results_df = (
        pd.DataFrame(results)
    )

    results_df = (
        results_df
        .sort_values(
            "best_val_loss"
        )
        .reset_index(
            drop=True
        )
    )

    summary_path = (
        RESULT_ROOT
        / "lookback_tuning_summary.csv"
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
        "Lookback Tuning Summary"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()