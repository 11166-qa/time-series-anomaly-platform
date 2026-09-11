from pathlib import Path
import time

import pandas as pd


from trainers.lstm_trainer import (
    train_lstm,
)

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



# ==========================================================
# 与 Electricity 保持一致的实验配置
# ==========================================================

BASE_CONFIG = {


    # ----------------------
    # sequence
    # ----------------------

    "lookback": 336,

    "pred_len": 24,



    # ----------------------
    # LSTM
    # ----------------------

    "hidden_size": 32,

    "num_layers": 1,



    # ----------------------
    # Transformer
    # iTransformer
    # ----------------------

    "d_model": 64,

    "nhead": 4,

    "dim_feedforward": 128,



    "dropout": 0.1,



    # ----------------------
    # training
    # ----------------------

    "batch_size": 128,

    "learning_rate": 0.001,

    "weight_decay": 1e-4,


    "max_epochs": 30,

    "early_stopping_patience": 8,

    "grad_clip": 1.0,


    "seed": 42,

}




def run_lstm():

    print("\n")
    print("=" * 70)
    print("Weather LSTM Baseline")
    print("=" * 70)


    config = BASE_CONFIG.copy()


    result_dir = (
        RESULT_ROOT
        /
        "LSTM"
        /
        "baseline"
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


    return {

        "model":
            "LSTM",

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





def run_transformer():


    print("\n")
    print("=" * 70)
    print("Weather Transformer Baseline")
    print("=" * 70)


    config = BASE_CONFIG.copy()


    result_dir = (
        RESULT_ROOT
        /
        "Transformer"
        /
        "baseline"
    )


    start_time = time.perf_counter()


    result = train_transformer(

        data_dir=DATA_DIR,

        result_dir=result_dir,

        config=config,

    )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    return {

        "model":
            "Transformer",

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





def run_itransformer():


    print("\n")
    print("=" * 70)
    print("Weather iTransformer Baseline")
    print("=" * 70)


    config = BASE_CONFIG.copy()



    result_dir = (
        RESULT_ROOT
        /
        "iTransformer"
        /
        "baseline"
    )


    start_time = time.perf_counter()


    result = train_itransformer(

        data_dir=DATA_DIR,

        result_dir=result_dir,

        config=config,

    )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    return {

        "model":
            "iTransformer",

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


    results = []


    # ======================================================
    # 三模型baseline
    # ======================================================


    results.append(
        run_lstm()
    )


    results.append(
        run_transformer()
    )


    results.append(
        run_itransformer()
    )



    df = pd.DataFrame(
        results
    )


    df = (
        df
        .sort_values(
            "best_val_loss"
        )
        .reset_index(
            drop=True
        )
    )


    summary_dir = (
        RESULT_ROOT
        /
        "baseline_summary"
    )


    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    save_path = (
        summary_dir
        /
        "weather_baseline_summary.csv"
    )


    df.to_csv(
        save_path,
        index=False,
    )



    print("\n")
    print("=" * 70)
    print("Weather Baseline Finished")
    print("=" * 70)


    print(
        df.to_string(
            index=False
        )
    )


    print(
        f"\nSaved to:\n{save_path}"
    )





if __name__ == "__main__":

    main()