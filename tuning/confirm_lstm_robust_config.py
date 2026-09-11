from pathlib import Path
import time

import numpy as np
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
    / "robust_config_confirmation"
)


BASE_CONFIG = {
    "lookback": 336,
    "pred_len": 24,

    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,

    "max_epochs": 30,
    "early_stopping_patience": 8,

    "grad_clip": 1.0,
}


CONFIGS = {
    "A": {
        "learning_rate": 2e-3,
        "weight_decay": 1e-2,
        "batch_size": 64,
    },

    "B": {
        "learning_rate": 2e-3,
        "weight_decay": 1e-3,
        "batch_size": 64,
    },

    "C": {
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "batch_size": 128,
    },
}


SEEDS = [
    42,
    123,
    2026,
    3407,
    7777,
]


def main():

    detailed_results = []

    for config_name, config_values in CONFIGS.items():

        for seed in SEEDS:

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"CONFIG = {config_name} | "
                f"SEED = {seed}"
            )

            print(
                "=" * 70
            )

            config = BASE_CONFIG.copy()
            config.update(config_values)

            config["seed"] = seed

            result_dir = (
                RESULT_ROOT
                / f"config_{config_name}"
                / f"seed_{seed}"
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

            detailed_results.append({
                "config":
                    config_name,

                "seed":
                    seed,

                "learning_rate":
                    config["learning_rate"],

                "weight_decay":
                    config["weight_decay"],

                "batch_size":
                    config["batch_size"],

                "best_val_loss":
                    result["best_val_loss"],

                "best_epoch":
                    result["best_epoch"],

                "epochs_run":
                    result["epochs_run"],

                "training_seconds":
                    elapsed_seconds,
            })

    detailed_df = pd.DataFrame(
        detailed_results
    )

    # ========================================================
    # 每套配置的 5-seed 稳定性
    # ========================================================

    summary_rows = []

    for config_name in CONFIGS:

        values = (
            detailed_df[
                detailed_df["config"]
                == config_name
            ]["best_val_loss"]
            .to_numpy()
        )

        config_values = (
            CONFIGS[config_name]
        )

        summary_rows.append({
            "config":
                config_name,

            "learning_rate":
                config_values[
                    "learning_rate"
                ],

            "weight_decay":
                config_values[
                    "weight_decay"
                ],

            "batch_size":
                config_values[
                    "batch_size"
                ],

            "mean_val_loss":
                np.mean(values),

            "std_val_loss":
                np.std(
                    values,
                    ddof=1,
                ),

            "min_val_loss":
                np.min(values),

            "max_val_loss":
                np.max(values),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = (
        summary_df
        .sort_values(
            [
                "mean_val_loss",
                "std_val_loss",
            ]
        )
        .reset_index(drop=True)
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_df.to_csv(
        RESULT_ROOT
        / "robust_config_seed_results.csv",
        index=False,
    )

    summary_df.to_csv(
        RESULT_ROOT
        / "robust_config_summary.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Robust Configuration Summary"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()