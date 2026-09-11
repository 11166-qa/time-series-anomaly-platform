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
    "Transformer"
    /
    "baseline"
)



CONFIG = {


    # =====================
    # Data
    # =====================

    "lookback": 336,

    "pred_len": 24,


    # =====================
    # Transformer
    # =====================

    "d_model": 32,

    "nhead": 4,

    "num_layers": 1,

    "dim_feedforward": 64,


    "dropout": 0.1,


    # =====================
    # Training
    # =====================

    "batch_size": 128,


    "learning_rate": 0.002,


    "weight_decay": 1e-5,


    "max_epochs": 30,


    "early_stopping_patience": 8,


    "grad_clip": 1.0,


    "seed": 42,

}



def main():


    print("=" * 70)

    print(
        "Electricity Transformer Baseline Training"
    )

    print("=" * 70)



    for k, v in CONFIG.items():

        print(
            f"{k}: {v}"
        )



    result_dir = (

        RESULT_ROOT

        /

        "seed_42"

    )



    start_time = time.perf_counter()



    result = train_transformer(

        data_dir=DATA_DIR,

        result_dir=result_dir,

        config=CONFIG,

    )



    elapsed = (

        time.perf_counter()

        -

        start_time

    )



    summary = {


        "model":

            "Transformer",


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

            elapsed,

    }



    df = pd.DataFrame(
        [summary]
    )



    RESULT_ROOT.mkdir(

        parents=True,

        exist_ok=True,

    )



    output_path = (

        RESULT_ROOT

        /

        "baseline_summary.csv"

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
        "Electricity Transformer Baseline Finished"
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
        f"\nSaved to:\n{output_path}"
    )



if __name__ == "__main__":

    main()