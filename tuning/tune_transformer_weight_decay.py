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
    / "weight_decay"
)


# 固定最佳结构和最佳学习率
BASE_CONFIG = {

    # =====================
    # Data
    # =====================
    "lookback": 336,
    "pred_len": 24,


    # =====================
    # Model
    # =====================
    "d_model": 32,
    "num_layers": 1,
    "nhead": 4,
    "dim_feedforward": 64,


    # =====================
    # Training
    # =====================
    "batch_size": 32,

    # 上一步LR tuning最佳
    "learning_rate": 0.002,

    # 当前固定
    "dropout": 0.1,

    "max_epochs": 30,

    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}



# weight decay搜索范围
WEIGHT_DECAYS = [
    0,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
]


def main():

    results = []


    for wd in WEIGHT_DECAYS:

        print("\n" + "=" * 70)
        print("Transformer Weight Decay Tuning")
        print(f"weight_decay = {wd}")
        print("=" * 70)



        config = BASE_CONFIG.copy()

        config["weight_decay"] = wd



        result_dir = (
            RESULT_ROOT
            /
            f"wd_{wd}"
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

            "weight_decay": wd,

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
        "weight_decay_tuning_summary.csv"
    )


    df.to_csv(
        save_path,
        index=False
    )



    print("\n")
    print("=" * 70)
    print("Weight Decay Tuning Summary")
    print("=" * 70)

    print(
        df.to_string(index=False)
    )


    print(
        f"\nSaved:\n{save_path}"
    )



if __name__ == "__main__":
    main()