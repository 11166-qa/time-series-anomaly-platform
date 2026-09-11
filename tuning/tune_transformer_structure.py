from pathlib import Path
import time

import pandas as pd







from trainers.transformer_trainer import (
    train_transformer,
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
    / "Transformer"
    / "tuning"
    / "structure"
)


BASE_CONFIG = {

    "lookback": 336,
    "pred_len": 24,

    "nhead": 4,

    "dropout": 0.1,

    "batch_size": 32,

    "learning_rate": 1e-3,

    "weight_decay": 1e-4,

    "max_epochs": 30,

    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


D_MODELS = [
    32,
    64,
    128,
]

NUM_LAYERS = [
    1,
    2,
]


def main():

    results = []

    trial = 0

    for d_model in D_MODELS:

        for num_layers in NUM_LAYERS:

            trial += 1

            dim_feedforward = (
                2 * d_model
            )

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Trial {trial}/6"
            )

            print(
                f"d_model = {d_model}"
            )

            print(
                f"num_layers = {num_layers}"
            )

            print(
                f"dim_feedforward = "
                f"{dim_feedforward}"
            )

            print(
                "=" * 70
            )

            # =================================================
            # 已经完成的 baseline 不重复训练
            # =================================================

            if (
                d_model == 64
                and num_layers == 2
            ):

                print(
                    "Using existing baseline result."
                )

                results.append({
                    "d_model":
                        64,

                    "num_layers":
                        2,

                    "nhead":
                        4,

                    "dim_feedforward":
                        128,

                    "best_val_loss":
                        0.411983,

                    "best_epoch":
                        20,

                    "epochs_run":
                        28,

                    "trainable_params":
                        78504,

                    "training_seconds":
                        8749.46,

                    "source":
                        "existing_baseline",
                })

                continue

            config = (
                BASE_CONFIG.copy()
            )

            config["d_model"] = (
                d_model
            )

            config["num_layers"] = (
                num_layers
            )

            config[
                "dim_feedforward"
            ] = dim_feedforward

            result_dir = (
                RESULT_ROOT
                / (
                    f"dmodel_{d_model}"
                    f"_layers_{num_layers}"
                )
            )

            start_time = (
                time.perf_counter()
            )

            result = train_transformer(
                data_dir=DATA_DIR,
                result_dir=result_dir,
                config=config,
            )

            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            results.append({
                "d_model":
                    d_model,

                "num_layers":
                    num_layers,

                "nhead":
                    4,

                "dim_feedforward":
                    dim_feedforward,

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

                "source":
                    "new_run",
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

    output_path = (
        RESULT_ROOT
        / "structure_tuning_summary.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Transformer Structure "
        "Tuning Summary"
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
        f"\nSaved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()