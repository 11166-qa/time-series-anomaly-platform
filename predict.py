import argparse

from pathlib import Path

import torch
import pandas as pd


from inference.predictor import (
    load_model,
    predict,
)



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        required=True
    )


    parser.add_argument(
        "--input",
        required=True
    )


    parser.add_argument(
        "--scaler",
        required=True
    )


    args = parser.parse_args()



    device=torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    model, checkpoint = load_model(

        args.checkpoint,

        device

    )


    result = predict(

        model,

        Path(args.input),

        Path(args.scaler),

        checkpoint,

        device

    )



    save_dir = Path(
        args.checkpoint
    ).parent

    # 判断输入数据是否为真实时间序列

    if "date" in pd.read_csv(args.input, nrows=1).columns:
        input_df = pd.read_csv(
            args.input,
            nrows=1
        )

        original_date = input_df["date"].iloc[0]

    # 工业数据使用采样索引
    if str(original_date).isdigit():

        time_column = "sample_index"

        timestamps = range(
            len(result["target"])
        )

    else:

        time_column = "date"

    pred_df = pd.DataFrame({

        time_column:

            timestamps,

        "target":

            result["target"],

        "prediction":

            result["prediction"],

    })


    pred_df["residual"] = (

        pred_df["target"]

        -

        pred_df["prediction"]

    ).abs()



    pred_df.to_csv(

        save_dir / "predictions.csv",

        index=False,

    )

    residual_df = pred_df[
        [
            time_column,
            "residual"
        ]
    ]


    residual_df.to_csv(

        save_dir / "residuals.csv",

        index=False,

    )


    print(
        "Prediction finished."
    )


    print(
        "Saved:",
        save_dir
    )



if __name__ == "__main__":

    main()