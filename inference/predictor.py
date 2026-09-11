from pathlib import Path

import torch
import numpy as np
import joblib


from datasets.time_series_dataset import TimeSeriesDataset

from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster
from models.itransformer import iTransformerForecaster



def load_model(
        checkpoint_path,
        device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


    model_name = checkpoint["model_name"]

    config = checkpoint["config"]

    input_size = checkpoint["input_size"]



    if model_name == "LSTM":

        model = LSTMForecaster(

            input_size=input_size,

            hidden_size=config["hidden_size"],

            num_layers=config["num_layers"],

            pred_len=config["pred_len"],

            output_size=input_size,

            dropout=config["dropout"],

        )


    elif model_name == "Transformer":

        model = TransformerForecaster(

            input_size=input_size,

            d_model=config["d_model"],

            nhead=config["nhead"],

            num_layers=config["num_layers"],

            dim_feedforward=config["dim_feedforward"],

            pred_len=config["pred_len"],

            output_size=input_size,

            dropout=config["dropout"],

        )


    elif model_name == "iTransformer":

        model = iTransformerForecaster(

            input_size=input_size,

            lookback=config["lookback"],

            pred_len=config["pred_len"],

            d_model=config["d_model"],

            nhead=config["nhead"],

            num_layers=config["num_layers"],

            dim_feedforward=config["dim_feedforward"],

            dropout=config["dropout"],

        )


    else:

        raise ValueError(
            f"Unsupported model {model_name}"
        )



    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.to(device)

    model.eval()


    return model, checkpoint





def predict(

        model,

        csv_path,

        scaler_path,

        checkpoint,

        device

):


    config = checkpoint["config"]



    dataset = TimeSeriesDataset(

        csv_path=csv_path,

        lookback=config["lookback"],

        pred_len=config["pred_len"],

    )


    loader = torch.utils.data.DataLoader(

        dataset,

        batch_size=128,

        shuffle=False,

        num_workers=0,

    )



    predictions = []

    targets = []

    timestamps = []



    with torch.inference_mode():


        for i,(x,y) in enumerate(loader):


            x = x.to(device)


            pred = model(x)


            predictions.append(
                pred.cpu().numpy()
            )


            targets.append(
                y.numpy()
            )



            # 保存预测区间时间

            batch_size = x.size(0)


            for j in range(batch_size):

                timestamps.extend(

                    dataset.get_target_timestamps(
                        i * 128 + j
                    )

                )



    predictions = np.concatenate(
        predictions,
        axis=0
    )


    targets = np.concatenate(
        targets,
        axis=0
    )


    timestamps = np.array(
        timestamps
    )



    # =================================================
    # inverse transform
    # =================================================

    scaler = joblib.load(
        scaler_path
    )


    n_samples = predictions.shape[0]


    pred_len = predictions.shape[1]


    n_features = predictions.shape[2]



    predictions_2d = predictions.reshape(
        -1,
        n_features
    )


    targets_2d = targets.reshape(
        -1,
        n_features
    )



    predictions_real = scaler.inverse_transform(
        predictions_2d
    )


    targets_real = scaler.inverse_transform(
        targets_2d
    )



    predictions_real = predictions_real.reshape(
        predictions.shape
    )


    targets_real = targets_real.reshape(
        targets.shape
    )



    # OT index = 最后一列

    ot_pred = predictions_real[:,:, -1]

    ot_true = targets_real[:,:, -1]



    return {

        "prediction":

            ot_pred.reshape(-1),

        "target":

            ot_true.reshape(-1),

        "timestamps":

            timestamps,

    }