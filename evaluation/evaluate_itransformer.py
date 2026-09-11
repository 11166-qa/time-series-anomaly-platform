from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)


from models.itransformer import (
    iTransformerForecaster,
)


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
    "ETTh1"
)


MODEL_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "ETTh1"
    /
    "iTransformer"
    /
    "seed"
)


RESULT_DIR = (
    PROJECT_ROOT
    /
    "results"
    /
    "ETTh1"
    /
    "iTransformer"
    /
    "final"
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



def inverse_transform_3d(
        data,
        scaler,
):

    original_shape = data.shape


    data_2d = data.reshape(
        -1,
        original_shape[-1],
    )


    restored = scaler.inverse_transform(
        data_2d
    )


    return restored.reshape(
        original_shape
    )



def load_model(
        checkpoint_path,
        device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )


    config = checkpoint["config"]


    model = iTransformerForecaster(

        input_size=7,

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
        checkpoint[
            "model_state_dict"
        ]
    )


    model.to(device)

    model.eval()


    return model, checkpoint



def predict(
        model,
        dataloader,
        device,
):

    predictions = []

    targets = []


    with torch.inference_mode():

        for x, y in dataloader:


            x = x.to(device)


            prediction = model(x)



            predictions.append(
                prediction
                .cpu()
                .numpy()
            )


            targets.append(
                y.numpy()
            )



    predictions = np.concatenate(
        predictions,
        axis=0,
    )


    targets = np.concatenate(
        targets,
        axis=0,
    )


    return predictions, targets



def calculate_variable_metrics(
        targets,
        predictions,
        feature_columns,
):

    rows = []


    for i, variable in enumerate(
            feature_columns
    ):


        y_true = (
            targets[:, :, i]
            .reshape(-1)
        )


        y_pred = (
            predictions[:, :, i]
            .reshape(-1)
        )


        metrics = regression_metrics(
            y_true,
            y_pred,
        )


        rows.append({

            "variable":
                variable,

            **metrics,

        })


    return pd.DataFrame(rows)



def main():


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 70)

    print(
        "ETTh1 iTransformer Final 5-Seed Evaluation"
    )

    print("=" * 70)


    print(
        f"Device: {device}"
    )



    scaler = joblib.load(
        SCALER_PATH
    )



    all_seed_metrics = []

    all_variable_metrics = []



    for seed in SEEDS:


        print(
            "\n"
            +
            "=" * 70
        )


        print(
            f"Evaluating seed = {seed}"
        )


        print(
            "=" * 70
        )



        checkpoint_path = (

            MODEL_ROOT

            /

            f"seed_{seed}"

            /

            "best_model.pt"

        )



        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"找不到模型: {checkpoint_path}"
            )



        model, checkpoint = load_model(
            checkpoint_path,
            device,
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



        test_loader = create_dataloader(

            dataset=test_dataset,


            batch_size=
                config["batch_size"],


            shuffle=False,


            num_workers=0,

        )



        pred_scaled, target_scaled = predict(

            model,

            test_loader,

            device,

        )



        predictions = inverse_transform_3d(

            pred_scaled,

            scaler,

        )


        targets = inverse_transform_3d(

            target_scaled,

            scaler,

        )



        variable_metrics = (
            calculate_variable_metrics(

                targets,

                predictions,

                test_dataset.feature_columns,

            )
        )



        variable_metrics.insert(

            0,

            "seed",

            seed,

        )



        all_variable_metrics.append(
            variable_metrics
        )



        macro = (
            variable_metrics[
                [
                    "R2",
                    "RMSE",
                    "MAE",
                    "MAPE",
                ]
            ]
            .mean()
        )



        seed_result = {

            "seed":

                seed,


            "R2":

                macro["R2"],


            "RMSE":

                macro["RMSE"],


            "MAE":

                macro["MAE"],


            "MAPE":

                macro["MAPE"],

        }



        all_seed_metrics.append(
            seed_result
        )



        print(
            f"Macro R2   : {macro['R2']:.6f}"
        )

        print(
            f"Macro RMSE : {macro['RMSE']:.6f}"
        )

        print(
            f"Macro MAE  : {macro['MAE']:.6f}"
        )

        print(
            f"Macro MAPE : {macro['MAPE']:.6f}"
        )



    # =====================================================
    # Summary
    # =====================================================


    seed_df = pd.DataFrame(
        all_seed_metrics
    )


    variable_df = pd.concat(
        all_variable_metrics,
        ignore_index=True,
    )



    summary_rows = []



    for metric in [

        "R2",

        "RMSE",

        "MAE",

        "MAPE",

    ]:


        values = (
            seed_df[metric]
            .to_numpy()
        )


        summary_rows.append({

            "metric":

                metric,


            "mean":

                np.mean(values),


            "std":

                np.std(
                    values,
                    ddof=1,
                ),


            "min":

                np.min(values),


            "max":

                np.max(values),

        })



    summary_df = pd.DataFrame(
        summary_rows
    )



    variable_summary = (

        variable_df

        .groupby("variable")[

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



    RESULT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )



    seed_df.to_csv(

        RESULT_DIR
        /
        "final_seed_metrics.csv",

        index=False,

    )


    variable_df.to_csv(

        RESULT_DIR
        /
        "final_variable_metrics_all_seeds.csv",

        index=False,

    )


    summary_df.to_csv(

        RESULT_DIR
        /
        "final_summary.csv",

        index=False,

    )


    variable_summary.to_csv(

        RESULT_DIR
        /
        "final_variable_summary.csv"

    )



    print(
        "\n"
        +
        "=" * 70
    )


    print(
        "Final 5-Seed Test Results"
    )


    print(
        "=" * 70
    )


    print(
        seed_df.to_string(
            index=False
        )
    )



    print(
        "\n"
        +
        "=" * 70
    )


    print(
        "Final Mean ± Std"
    )


    print(
        "=" * 70
    )



    for _, row in summary_df.iterrows():

        print(

            f"{row['metric']:>5} : "

            f"{row['mean']:.6f}"

            " ± "

            f"{row['std']:.6f}"

        )



    print(
        f"\nResults saved to:\n{RESULT_DIR}"
    )



if __name__ == "__main__":

    main()