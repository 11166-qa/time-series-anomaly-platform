from pathlib import Path
import json

import pandas as pd
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)



def evaluate(
        prediction_file
):

    df = pd.read_csv(
        prediction_file
    )


    y_true = df["target"].values

    y_pred = df["prediction"].values



    r2 = r2_score(
        y_true,
        y_pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    # 避免除0
    mape = np.mean(
        np.abs(
            (y_true-y_pred)
            /
            (np.abs(y_true)+1e-8)
        )
    ) * 100



    metrics = {

        "R2":
            float(r2),

        "RMSE":
            float(rmse),

        "MAE":
            float(mae),

        "MAPE":
            float(mape),

        "samples":
            int(len(df))

    }


    return metrics





if __name__ == "__main__":


    prediction_file = Path(

        "results/Weather/iTransformer/product/predictions_merged.csv"

    )


    metrics = evaluate(
        prediction_file
    )


    print("="*60)

    print("Evaluation Result")

    print("="*60)


    for k,v in metrics.items():

        print(
            f"{k}: {v}"
        )



    output_file = (

        prediction_file.parent

        /

        "metrics.json"

    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    print()

    print(
        "Saved:",
        output_file
    )