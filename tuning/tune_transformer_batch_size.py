from pathlib import Path
import time

import pandas as pd

from trainers.transformer_trainer import train_transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    / "batch_size"
)


BASE_CONFIG = {

    # Data
    "lookback": 336,
    "pred_len": 24,


    # Model
    "d_model": 32,
    "num_layers": 1,
    "nhead": 4,
    "dim_feedforward": 64,


    # Training
    "learning_rate": 0.002,

    "weight_decay": 1e-5,

    "dropout": 0.1,


    "max_epochs": 30,

    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


BATCH_SIZES = [
    16,
    32,
    64,
    128,
]


def main():

    results = []


    for batch_size in BATCH_SIZES:

        print("\n" + "=" * 70)
        print("Transformer Batch Size Tuning")
        print(f"batch_size = {batch_size}")
        print("=" * 70)


        config = BASE_CONFIG.copy()

        config["batch_size"] = batch_size


        result_dir = (
            RESULT_ROOT
            /
            f"batch_{batch_size}"
        )


        start = time.perf_counter()


        result = train_transformer(
            data_dir=DATA_DIR,
            result_dir=result_dir,
            config=config
        )


        elapsed = (
            time.perf_counter()
            -
            start
        )


        results.append({

            "batch_size": batch_size,

            "best_val_loss":
                result["best_val_loss"],

            "best_epoch":
                result["best_epoch"],

            "epochs_run":
                result["epochs_run"],

            "trainable_params":
                result["trainable_params"],

            "training_seconds":
                elapsed

        })


    df = pd.DataFrame(results)


    df = (
        df
        .sort_values(
            "best_val_loss"
        )
        .reset_index(drop=True)
    )


    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    save_path = (
        RESULT_ROOT
        /
        "batch_size_tuning_summary.csv"
    )


    df.to_csv(
        save_path,
        index=False
    )


    print("\n")
    print("=" * 70)
    print("Batch Size Tuning Summary")
    print("=" * 70)

    print(
        df.to_string(index=False)
    )


    print(
        f"\nSaved:\n{save_path}"
    )


if __name__ == "__main__":
    main()