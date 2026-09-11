from pathlib import Path

import joblib
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


from utils.metrics import regression_metrics



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


RESULT_DIR = (
    RESULT_ROOT
    /
    "final_evaluation"
)


SCALER_PATH = (
    DATA_DIR
    /
    "scaler.joblib"
)


SEEDS = [
    42,
    123,
    2026,
    3407,
    7777,
]


MODELS = [
    "LSTM",
    "Transformer",
    "iTransformer",
]



def inverse_transform_3d(
        data,
        scaler,
):

    shape = data.shape

    data_2d = data.reshape(
        -1,
        shape[-1],
    )

    restored = scaler.inverse_transform(
        data_2d
    )

    return restored.reshape(shape)



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


    else:

        raise ValueError(
            model_name
        )



    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.to(device)

    model.eval()


    return model, checkpoint




def predict(
        model,
        loader,
        device,
        model_name,
        pred_len,
):

    predictions = []
    targets = []


    with torch.inference_mode():

        for x, y in loader:

            x = x.to(device)

            pred = model(x)


            # LSTM / Transformer
            # [B,7704]
            # ->
            # [B,24,321]

            if model_name in [
                "LSTM",
                "Transformer",
            ]:

                pred = pred.reshape(
                    pred.shape[0],
                    pred_len,
                    -1,
                )


            predictions.append(
                pred.cpu().numpy()
            )


            targets.append(
                y.numpy()
            )


    return (
        np.concatenate(
            predictions,
            axis=0,
        ),

        np.concatenate(
            targets,
            axis=0,
        )
    )




def calculate_macro_metrics(
        predictions,
        targets,
):


    metrics_list = []


    feature_num = predictions.shape[-1]


    for i in range(feature_num):


        y_true = (

            targets[:, :, i]

            .reshape(-1)

        )


        y_pred = (

            predictions[:, :, i]

            .reshape(-1)

        )


        metric = regression_metrics(

            y_true,

            y_pred,

        )


        metrics_list.append(metric)



    df = pd.DataFrame(
        metrics_list
    )


    return {

        "R2":
            df["R2"].mean(),

        "RMSE":
            df["RMSE"].mean(),

        "MAE":
            df["MAE"].mean(),

        "MAPE":
            df["MAPE"].mean(),

    }




def evaluate_one_model(
        model_name,
        device,
        scaler,
):


    print("\n")
    print("=" * 70)
    print(
        f"Evaluating {model_name}"
    )
    print("=" * 70)



    results = []



    for seed in SEEDS:


        print(
            f"\nSeed = {seed}"
        )


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



        if not checkpoint_path.exists():

            raise FileNotFoundError(

                checkpoint_path

            )



        checkpoint = torch.load(

            checkpoint_path,

            map_location="cpu",

            weights_only=False,

        )


        config = checkpoint["config"]



        test_dataset = TimeSeriesDataset(

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


        input_size = (
            test_dataset.num_features
        )



        model, _ = load_model(

            model_name,

            checkpoint_path,

            device,

            input_size,

        )



        loader = create_dataloader(

            test_dataset,

            batch_size=
                config["batch_size"],

            shuffle=False,

            num_workers=0,

        )



        pred_scaled, true_scaled = predict(

            model,

            loader,

            device,

            model_name,

            config["pred_len"],

        )



        predictions = inverse_transform_3d(

            pred_scaled,

            scaler,

        )


        targets = inverse_transform_3d(

            true_scaled,

            scaler,

        )
        print(
            model_name,
            "prediction range:",
            predictions.min(),
            predictions.max()
        )

        print(
            model_name,
            "target range:",
            targets.min(),
            targets.max()
        )



        metric = calculate_macro_metrics(

            predictions,

            targets,

        )


        metric["model"] = model_name

        metric["seed"] = seed



        results.append(metric)



        print(metric)



    return pd.DataFrame(results)




def main():


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(
        f"Device: {device}"
    )



    scaler = joblib.load(

        SCALER_PATH

    )



    all_results = []



    for model_name in MODELS:


        df = evaluate_one_model(

            model_name,

            device,

            scaler,

        )


        all_results.append(df)



    results = pd.concat(

        all_results,

        ignore_index=True,

    )



    RESULT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )



    results.to_csv(

        RESULT_DIR
        /
        "all_seed_results.csv",

        index=False,

    )



    summary = (

        results

        .groupby("model")

        [

            [

                "R2",

                "RMSE",

                "MAE",

                "MAPE",

            ]

        ]

        .agg(

            [

                "mean",

                "std",

            ]

        )

    )



    summary.to_csv(

        RESULT_DIR
        /
        "final_summary.csv"

    )



    print("\n")
    print("=" * 70)
    print(
        "Electricity Final Summary"
    )
    print("=" * 70)


    print(summary)



    print(
        f"\nSaved to:\n{RESULT_DIR}"
    )



if __name__ == "__main__":

    main()