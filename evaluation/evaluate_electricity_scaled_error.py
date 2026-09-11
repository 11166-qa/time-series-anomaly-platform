from pathlib import Path

import numpy as np
import pandas as pd
import torch


from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)


from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster
from models.itransformer import iTransformerForecaster



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
)


MODELS = [
    "LSTM",
    "Transformer",
    "iTransformer",
]


SEEDS = [
    42,
    123,
    2026,
    3407,
    7777,
]



def load_model(
        model_name,
        checkpoint_path,
        device,
        input_size,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )


    config = checkpoint["config"]


    if model_name == "LSTM":


        model = LSTMForecaster(

            input_size=input_size,

            hidden_size=
                config["hidden_size"],

            num_layers=
                config["num_layers"],

            pred_len=
                config["pred_len"],

            output_size=input_size,

            dropout=
                config["dropout"],

        )


    elif model_name == "Transformer":


        model = TransformerForecaster(

            input_size=input_size,

            d_model=
                config["d_model"],

            nhead=
                config["nhead"],

            num_layers=
                config["num_layers"],

            dim_feedforward=
                config["dim_feedforward"],

            pred_len=
                config["pred_len"],

            dropout=
                config["dropout"],

        )


    elif model_name == "iTransformer":


        model = iTransformerForecaster(

            input_size=input_size,

            lookback=
                config["lookback"],

            pred_len=
                config["pred_len"],

            d_model=
                config["d_model"],

            nhead=
                config["nhead"],

            num_layers=
                config["num_layers"],

            dim_feedforward=
                config["dim_feedforward"],

            dropout=
                config["dropout"],

        )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.to(device)

    model.eval()


    return model, config



def evaluate_seed(
        model,
        loader,
        device,
):

    mse_list = []

    rmse_list = []


    with torch.inference_mode():


        for x,y in loader:


            x = x.to(device)


            pred = model(x)


            mse = torch.mean(
                (pred.cpu()-y)**2
            )


            rmse = torch.sqrt(
                mse
            )


            mse_list.append(
                mse.item()
            )


            rmse_list.append(
                rmse.item()
            )


    return {

        "MSE":
            np.mean(mse_list),

        "RMSE":
            np.mean(rmse_list),

    }




def main():


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Device: {device}"
    )


    results = []



    for model_name in MODELS:


        print("\n")
        print("="*70)
        print(model_name)
        print("="*70)


        for seed in SEEDS:


            checkpoint_path = (

                RESULT_ROOT

                /

                model_name

                /

                "seed"

                /

                f"seed_{seed}"

                /

                "best_model.pt"

            )


            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )


            config = checkpoint["config"]


            dataset = TimeSeriesDataset(

                csv_path=
                    DATA_DIR
                    /
                    "test.csv",

                context_csv_path=
                    DATA_DIR
                    /
                    "val.csv",

                lookback=
                    config["lookback"],

                pred_len=
                    config["pred_len"],

            )


            loader = create_dataloader(

                dataset,

                batch_size=
                    config["batch_size"],

                shuffle=False,

                num_workers=0,

            )


            model,_ = load_model(

                model_name,

                checkpoint_path,

                device,

                dataset.num_features,

            )


            metrics = evaluate_seed(

                model,

                loader,

                device,

            )


            metrics["model"] = model_name

            metrics["seed"] = seed


            results.append(metrics)


            print(
                metrics
            )



    df = pd.DataFrame(results)


    print("\n")
    print("="*70)
    print("Mean ± Std")
    print("="*70)


    print(
        df.groupby("model")
        [
            [
                "MSE",
                "RMSE"
            ]
        ]
        .agg(
            [
                "mean",
                "std"
            ]
        )
    )



    output = (
        RESULT_ROOT
        /
        "scaled_error_summary.csv"
    )


    df.to_csv(
        output,
        index=False,
    )


    print(
        f"\nSaved:\n{output}"
    )



if __name__ == "__main__":

    main()