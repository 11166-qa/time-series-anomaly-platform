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



    pred_df = pd.DataFrame({

        "date":

            result["timestamps"],

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
            "date",
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