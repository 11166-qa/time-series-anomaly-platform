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
    /
    "data"
    /
    "processed"
    /
    "Electricity"
)


RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "Electricity"
    /
    "LSTM"
    /
    "seed"
)


SEEDS = [
    42,
    123,
    2026,
    3407,
    7777,
]


BASE_CONFIG = {

    "lookback": 336,

    "pred_len": 24,


    "hidden_size": 32,

    "num_layers": 1,

    "dropout": 0.1,


    "batch_size": 64,


    "learning_rate": 0.002,


    "weight_decay": 0.01,


    "max_epochs": 30,


    "early_stopping_patience": 8,


    "grad_clip": 1.0,

}



def main():

    results = []


    for seed in SEEDS:


        print(
            "\n"
            +
            "=" * 70
        )

        print(
            f"Electricity LSTM Seed = {seed}"
        )

        print(
            "=" * 70
        )


        config = BASE_CONFIG.copy()

        config["seed"] = seed


        result_dir = (
            RESULT_ROOT
            /
            f"seed_{seed}"
        )


        start_time = time.perf_counter()


        result = train_lstm(
            data_dir=DATA_DIR,
            result_dir=result_dir,
            config=config,
        )


        elapsed = (
            time.perf_counter()
            -
            start_time
        )


        results.append({

            "seed":
                seed,

            "best_val_loss":
                result["best_val_loss"],

            "best_epoch":
                result["best_epoch"],

            "epochs_run":
                result["epochs_run"],

            "trainable_params":
                result["trainable_params"],

            "training_seconds":
                elapsed,

        })


    df = pd.DataFrame(results)


    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path = (
        RESULT_ROOT
        /
        "seed_stability_summary.csv"
    )


    df.to_csv(
        output_path,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "Electricity LSTM Seed Stability Summary"
    )

    print(
        "=" * 70
    )

    print(
        df.to_string(
            index=False
        )
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "Mean ± Std"
    )

    print(
        "=" * 70
    )


    print(
        f"mean_val_loss: "
        f"{df['best_val_loss'].mean():.6f}"
    )

    print(
        f"std_val_loss: "
        f"{df['best_val_loss'].std():.6f}"
    )

    print(
        f"min_val_loss: "
        f"{df['best_val_loss'].min():.6f}"
    )

    print(
        f"max_val_loss: "
        f"{df['best_val_loss'].max():.6f}"
    )


    print(
        f"\nSaved to:\n{output_path}"
    )



if __name__ == "__main__":

    main()