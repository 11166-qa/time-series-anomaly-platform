from pathlib import Path
import time

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
    / "capacity"
)


BASE_CONFIG = {
    "lookback": 336,
    "pred_len": 24,

    "batch_size": 64,

    "dropout": 0.2,

    "learning_rate": 1e-3,
    "weight_decay": 1e-4,

    "max_epochs": 30,
    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


HIDDEN_SIZES = [
    32,
    64,
    128,
]

NUM_LAYERS_VALUES = [
    1,
    2,
]


def main():

    results = []

    trial = 0

    for hidden_size in HIDDEN_SIZES:

        for num_layers in NUM_LAYERS_VALUES:

            trial += 1

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Trial {trial} | "
                f"hidden_size={hidden_size} | "
                f"num_layers={num_layers}"
            )

            print(
                "=" * 70
            )

            config = (
                BASE_CONFIG.copy()
            )

            config[
                "hidden_size"
            ] = hidden_size

            config[
                "num_layers"
            ] = num_layers

            result_dir = (
                RESULT_ROOT
                / (
                    f"hidden_{hidden_size}"
                    f"_layers_{num_layers}"
                )
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
                "hidden_size":
                    hidden_size,

                "num_layers":
                    num_layers,

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

    results_df = pd.DataFrame(
        results
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

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        RESULT_ROOT
        / "capacity_tuning_summary.csv"
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
        "LSTM Capacity Tuning Summary"
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