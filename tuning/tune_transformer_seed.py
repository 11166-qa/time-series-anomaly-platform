from pathlib import Path
import time
import random

import numpy as np
import torch
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
    / "seed"
)



# ==========================================================
# 固定最终Transformer配置
# ==========================================================

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

    "batch_size": 128,


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



def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)



    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



def main():

    results = []


    for seed in SEEDS:


        print("\n" + "=" * 70)

        print("Transformer Seed Stability Test")

        print(f"seed = {seed}")

        print("=" * 70)



        set_seed(seed)



        config = BASE_CONFIG.copy()

        config["seed"] = seed



        result_dir = (
            RESULT_ROOT
            /
            f"seed_{seed}"
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

            "seed": seed,

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



    summary = pd.DataFrame({

        "metric": [
            "mean_val_loss",
            "std_val_loss",
            "min_val_loss",
            "max_val_loss"
        ],

        "value": [
            df["best_val_loss"].mean(),
            df["best_val_loss"].std(),
            df["best_val_loss"].min(),
            df["best_val_loss"].max()
        ]

    })



    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )



    df.to_csv(
        RESULT_ROOT / "seed_results.csv",
        index=False
    )


    summary.to_csv(
        RESULT_ROOT / "seed_summary.csv",
        index=False
    )



    print("\n")
    print("=" * 70)

    print("Seed Stability Results")

    print("=" * 70)

    print(
        df.to_string(index=False)
    )


    print("\n")

    print(
        summary.to_string(index=False)
    )



if __name__ == "__main__":

    main()