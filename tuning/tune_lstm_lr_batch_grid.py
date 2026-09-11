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
    / "lr_batch_grid"
)


BASE_CONFIG = {
    "lookback": 336,
    "pred_len": 24,

    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,

    "weight_decay": 1e-2,

    "max_epochs": 30,
    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


LEARNING_RATES = [
    1e-3,
    2e-3,
    3e-3,
]

BATCH_SIZES = [
    32,
    64,
    128,
]


def main():

    results = []

    trial = 0

    for learning_rate in LEARNING_RATES:

        for batch_size in BATCH_SIZES:

            trial += 1

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Trial {trial}/9 | "
                f"LR={learning_rate} | "
                f"Batch={batch_size}"
            )

            print(
                "=" * 70
            )

            config = BASE_CONFIG.copy()

            config["learning_rate"] = (
                learning_rate
            )

            config["batch_size"] = (
                batch_size
            )

            lr_name = (
                f"{learning_rate:.0e}"
                .replace("-", "m")
            )

            result_dir = (
                RESULT_ROOT
                / (
                    f"lr_{lr_name}"
                    f"_batch_{batch_size}"
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
                "learning_rate":
                    learning_rate,

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

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
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
        / "lr_batch_grid_summary.csv"
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
        "LR x Batch Grid Search Summary"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # 额外生成矩阵形式，方便观察交互
    pivot = results_df.pivot(
        index="learning_rate",
        columns="batch_size",
        values="best_val_loss",
    )

    pivot_path = (
        RESULT_ROOT
        / "lr_batch_grid_matrix.csv"
    )

    pivot.to_csv(
        pivot_path
    )

    print(
        "\nValidation Loss Matrix:"
    )

    print(pivot)

    print(
        f"\nSummary saved to:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()