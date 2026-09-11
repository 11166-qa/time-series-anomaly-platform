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
    / "weight_decay_final"
)

BASE_CONFIG = {
    "lookback": 336,
    "pred_len": 24,

    "batch_size": 64,

    "hidden_size": 32,
    "num_layers": 1,

    # 单层LSTM内部dropout不生效
    "dropout": 0.0,

    # 当前最优学习率
    "learning_rate": 2e-3,

    "max_epochs": 30,

    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


WEIGHT_DECAY_VALUES = [
    1e-2,
    2e-2,
    5e-2,
    1e-1,
]

def main():

    results = []

    for weight_decay in WEIGHT_DECAY_VALUES:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"WEIGHT DECAY = {weight_decay}"
        )

        print(
            "=" * 70
        )

        config = BASE_CONFIG.copy()

        config["weight_decay"] = (
            weight_decay
        )

        wd_name = (
            f"{weight_decay:.0e}"
            .replace("-", "m")
        )

        result_dir = (
            RESULT_ROOT
            / f"wd_{wd_name}"
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
            "weight_decay":
                weight_decay,

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
            / "weight_decay_final_summary.csv"
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
        "Weight Decay Tuning Summary"
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