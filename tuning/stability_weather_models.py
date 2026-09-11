from pathlib import Path
import time

import numpy as np
import pandas as pd


from trainers.transformer_trainer import (
    train_transformer,
)

from trainers.itransformer_trainer import (
    train_itransformer,
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
    "Weather"
)



RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "Weather"
)



SEEDS = [
    42,
    123,
    2026,
    3407,
    7777,
]



BASE_CONFIG = {


    "lookback":336,

    "pred_len":24,


    # Transformer
    "d_model":64,

    "nhead":4,

    "num_layers":1,

    "dim_feedforward":128,


    # common
    "dropout":0.1,


    "batch_size":128,

    "learning_rate":0.001,

    "weight_decay":1e-4,


    "max_epochs":30,

    "early_stopping_patience":8,

    "grad_clip":1.0,

}




MODELS = [

    "Transformer",

    "iTransformer",

]




def run_one(
        model_name,
        seed,
):


    config = BASE_CONFIG.copy()

    config["seed"] = seed



    result_dir = (

        RESULT_ROOT

        /

        model_name

        /

        "stability"

        /

        f"seed_{seed}"

    )



    print("\n")
    print("="*70)

    print(
        f"{model_name} | seed={seed}"
    )

    print("="*70)



    start = time.perf_counter()



    if model_name == "Transformer":


        result = train_transformer(

            data_dir=DATA_DIR,

            result_dir=result_dir,

            config=config,

        )


    elif model_name == "iTransformer":


        result = train_itransformer(

            data_dir=DATA_DIR,

            result_dir=result_dir,

            config=config,

        )


    else:

        raise ValueError(
            model_name
        )



    elapsed = (
        time.perf_counter()
        -
        start
    )



    return {


        "model":
            model_name,


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

    }




def main():


    results=[]



    for model_name in MODELS:


        for seed in SEEDS:


            result = run_one(

                model_name,

                seed,

            )


            results.append(
                result
            )



    df = pd.DataFrame(
        results
    )


    output_dir = (

        RESULT_ROOT

        /

        "stability_summary"

    )


    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    df.to_csv(

        output_dir
        /
        "weather_transformer_itransformer_stability_all.csv",

        index=False,

    )



    summary = (

        df

        .groupby(
            "model"
        )

        [
            "best_val_loss"
        ]

        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )

    )



    summary.to_csv(

        output_dir
        /
        "weather_transformer_itransformer_stability_summary.csv"

    )



    print("\n")
    print("="*70)

    print(
        "Weather Stability Summary"
    )

    print("="*70)


    print(summary)



if __name__ == "__main__":

    main()