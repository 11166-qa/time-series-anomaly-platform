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
    / "seed_validation"
)


FINAL_CONFIG = {
    "lookback": 336,
    "pred_len": 24,

    "batch_size": 64,

    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,

    "learning_rate": 2e-3,
    "weight_decay": 1e-2,

    "max_epochs": 30,
    "early_stopping_patience": 8,

    "grad_clip": 1.0,
}


SEEDS = [
    42,
    123,
    2026,
    3407,
    7777,
]


def main():

    results = []

    for seed in SEEDS:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"SEED = {seed}"
        )

        print(
            "=" * 70
        )

        config = (
            FINAL_CONFIG.copy()
        )

        config["seed"] = seed

        result_dir = (
            RESULT_ROOT
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

        results.append({
            "seed":
                seed,

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

    # ========================================================
    # 稳定性统计
    # ========================================================

    val_losses = (
        results_df[
            "best_val_loss"
        ]
        .to_numpy()
    )

    summary_df = pd.DataFrame([
        {
            "num_seeds":
                len(SEEDS),

            "mean_val_loss":
                np.mean(
                    val_losses
                ),

            # 样本标准差
            "std_val_loss":
                np.std(
                    val_losses,
                    ddof=1,
                ),

            "min_val_loss":
                np.min(
                    val_losses
                ),

            "max_val_loss":
                np.max(
                    val_losses
                ),
        }
    ])

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        RESULT_ROOT
        / "seed_results.csv"
    )

    summary_path = (
        RESULT_ROOT
        / "seed_summary.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Seed Validation Results"
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
        "\n"
        + "=" * 70
    )

    print(
        "Seed Stability Summary"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULT_ROOT}"
    )


if __name__ == "__main__":
    main()