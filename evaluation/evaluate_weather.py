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
    "Weather"
)


RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "Weather"
)



SCALER_PATH = (
    DATA_DIR
    /
    "scaler.joblib"
)



# =====================================================
# 最终模型路径
# =====================================================

MODEL_PATHS = {


    "LSTM":

    (
        RESULT_ROOT
        /
        "LSTM"
        /
        "baseline"
        /
        "best_model.pt"
    ),



    # 根据稳定性结果修改
    # 替换成你的最低val loss对应seed

    "Transformer":

(
    RESULT_ROOT
    /
    "Transformer"
    /
    "stability"
    /
    "seed_2026"
    /
    "best_model.pt"
),



    "iTransformer":

(
    RESULT_ROOT
    /
    "iTransformer"
    /
    "stability"
    /
    "seed_3407"
    /
    "best_model.pt"
),


}



RESULT_DIR = (

    RESULT_ROOT

    /

    "final"

)



def inverse_transform_3d(
        data,
        scaler,
):

    shape = data.shape


    data2d = (
        data
        .reshape(
            -1,
            shape[-1]
        )
    )


    restored = (
        scaler
        .inverse_transform(
            data2d
        )
    )


    return restored.reshape(
        shape
    )




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
        checkpoint["model_state_dict"]
    )


    model.to(device)

    model.eval()


    return model, checkpoint





def predict(
        model,
        loader,
        device,
):


    preds = []

    trues = []



    with torch.inference_mode():


        for x,y in loader:


            x = x.to(device)


            pred = model(x)



            preds.append(
                pred.cpu()
                .numpy()
            )


            trues.append(
                y.numpy()
            )



    return (

        np.concatenate(
            preds,
            axis=0
        ),

        np.concatenate(
            trues,
            axis=0
        )

    )





def evaluate_model(
        model_name,
        model_path,
        scaler,
        device,
):


    print("\n")
    print("="*70)
    print(model_name)
    print("="*70)



    checkpoint = torch.load(
        model_path,
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

        model_path,

        device,

        dataset.num_features,

    )



    pred_scaled,true_scaled = predict(

        model,

        loader,

        device,

    )



    pred = inverse_transform_3d(

        pred_scaled,

        scaler,

    )


    true = inverse_transform_3d(

        true_scaled,

        scaler,

    )



    print(
        "Prediction range:",
        pred.min(),
        pred.max()
    )


    print(
        "Target range:",
        true.min(),
        true.max()
    )



    metrics = regression_metrics(

        true.reshape(-1),

        pred.reshape(-1),

    )


    metrics["model"] = model_name



    print(metrics)



    return metrics





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



    results = []



    for model_name,model_path in MODEL_PATHS.items():


        if not model_path.exists():

            raise FileNotFoundError(
                f"模型不存在:\n{model_path}"
            )


        result = evaluate_model(

            model_name,

            model_path,

            scaler,

            device,

        )


        results.append(result)



    df = pd.DataFrame(
        results
    )


    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )



    df.to_csv(

        RESULT_DIR
        /
        "weather_final_metrics.csv",

        index=False,

    )



    print("\n")
    print("="*70)
    print("Weather Final Summary")
    print("="*70)


    print(
        df.to_string(
            index=False
        )
    )



    print(
        "\nSaved to:"
    )

    print(
        RESULT_DIR
        /
        "weather_final_metrics.csv"
    )




if __name__ == "__main__":

    main()