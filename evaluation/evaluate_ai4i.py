from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)



PROJECT_ROOT = Path(__file__).resolve().parent.parent


RESULT_DIR = (
    PROJECT_ROOT
    /
    "results"
    /
    "AI4I"
)



MODELS = [

    "LSTM",

    "Transformer",

    "iTransformer"

]





def calculate_metrics(df):


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


    mape = np.mean(

        np.abs(

            (y_true-y_pred)

            /

            (np.abs(y_true)+1e-8)

        )

    ) * 100



    return {

        "R2": r2,

        "RMSE": rmse,

        "MAE": mae,

        "MAPE": mape

    }





def main():


    results = []



    for model in MODELS:


        file = (

            RESULT_DIR

            /

            model

            /

            "case_study"

            /

            "predictions.csv"

        )


        if not file.exists():

            print(
                "Missing:",
                file
            )

            continue



        df = pd.read_csv(
            file
        )


        metrics = calculate_metrics(
            df
        )


        metrics["model"] = model


        results.append(
            metrics
        )



    result_df = pd.DataFrame(
        results
    )


    result_df = result_df[

        [

            "model",

            "R2",

            "RMSE",

            "MAE",

            "MAPE"

        ]

    ]



    print("="*70)

    print("AI4I Forecasting Performance")

    print("="*70)


    print(
        result_df.to_string(
            index=False
        )
    )



    output = (

        RESULT_DIR

        /

        "AI4I_forecasting_results.csv"

    )


    result_df.to_csv(

        output,

        index=False

    )


    print()

    print(
        "Saved:",
        output
    )





if __name__ == "__main__":

    main()