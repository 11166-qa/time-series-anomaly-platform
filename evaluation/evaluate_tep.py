from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)



# ======================================================
# Path
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


RESULT_DIR = (
    PROJECT_ROOT
    /
    "results"
    /
    "TEP"
)



MODELS = [

    "LSTM",

    "Transformer",

    "iTransformer"

]





# ======================================================
# Metrics
# ======================================================

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



    return {

        "R2":

            r2,


        "RMSE":

            rmse,


        "MAE":

            mae

    }





# ======================================================
# Main
# ======================================================

def main():


    results = []



    for model in MODELS:


        prediction_file = (

            RESULT_DIR

            /

            model

            /

            "case_study"

            /

            "predictions.csv"

        )


        if not prediction_file.exists():

            print(

                "Missing:",

                prediction_file

            )

            continue



        df = pd.read_csv(

            prediction_file

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

            "MAE"

        ]

    ]



    print("=" * 70)

    print(

        "TEP Forecasting Performance"

    )

    print("=" * 70)



    print(

        result_df.to_string(

            index=False

        )

    )



    output_file = (

        RESULT_DIR

        /

        "TEP_forecasting_results.csv"

    )



    result_df.to_csv(

        output_file,

        index=False

    )



    print()

    print(

        "Saved:",

        output_file

    )





if __name__ == "__main__":

    main()