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
    / "learning_rate_refine"
)


BASE_CONFIG = {
    "lookback": 336,
    "pred_len": 24,

    "batch_size": 64,

    "hidden_size": 32,
    "num_layers": 1,

    # 单层LSTM内部dropout不生效
    "dropout": 0.0,

    "weight_decay": 1e-4,

    "max_epochs": 30,
    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


LEARNING_RATES = [
    2e-3,
    3e-3,
    4e-3,
    5e-3,
]


def main():

    results = []

    for lr in LEARNING_RATES:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"LEARNING RATE = {lr}"
        )

        print(
            "=" * 70
        )

        config = BASE_CONFIG.copy()

        config[
            "learning_rate"
        ] = lr

        lr_name = (
            f"{lr:.0e}"
            .replace("-", "m")
        )

        result_dir = (
            RESULT_ROOT
            / f"lr_{lr_name}"
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
            "learning_rate":
                lr,

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
        / "learning_rate_refine_summary.csv"
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
        "Learning Rate Refine Summary"
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