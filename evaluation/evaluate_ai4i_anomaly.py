from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)



PROJECT_ROOT = Path(__file__).resolve().parent.parent


LABEL_FILE = (

    PROJECT_ROOT

    /

    "data"

    /

    "processed"

    /

    "AI4I"

    /

    "labels.csv"

)



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



def detect_3sigma(
        residual
):

    threshold = (

        residual.mean()

        +

        3 * residual.std()

    )


    return (

        residual > threshold

    ).astype(int)





def detect_iqr(
        residual
):

    q1 = residual.quantile(
        0.25
    )


    q3 = residual.quantile(
        0.75
    )


    iqr = q3-q1


    threshold = (

        q3

        +

        1.5*iqr

    )


    return (

        residual > threshold

    ).astype(int)





def detect_iforest(
        residual
):

    from sklearn.ensemble import IsolationForest


    model = IsolationForest(

        contamination=0.05,

        random_state=42

    )


    pred = model.fit_predict(

        residual.values.reshape(-1,1)

    )


    return (

        pred == -1

    ).astype(int)





def calculate_metrics(
        y_true,
        y_pred
):


    return {

        "Precision":

            precision_score(

                y_true,

                y_pred,

                zero_division=0

            ),


        "Recall":

            recall_score(

                y_true,

                y_pred,

                zero_division=0

            ),


        "F1":

            f1_score(

                y_true,

                y_pred,

                zero_division=0

            )

    }





def main():


    labels = pd.read_csv(
        LABEL_FILE
    )


    all_results = []



    for model in MODELS:


        residual_file = (

            RESULT_DIR

            /

            model

            /

            "case_study"

            /

            "residuals.csv"

        )


        if not residual_file.exists():

            print(
                "Missing:",
                residual_file
            )

            continue



        residual_df = pd.read_csv(

            residual_file

        )


        df = residual_df.merge(

            labels,

            on="date",

            how="inner"

        )



        print(

            model,

            "matched samples:",

            len(df)

        )



        y_true = df["label"]



        methods = {

            "3sigma":

                detect_3sigma(

                    df["residual"]

                ),


            "IQR":

                detect_iqr(

                    df["residual"]

                ),


            "IsolationForest":

                detect_iforest(

                    df["residual"]

                )

        }



        for name,pred in methods.items():


            metrics = calculate_metrics(

                y_true,

                pred

            )


            metrics.update({

                "Model":

                    model,


                "Method":

                    name

            })


            all_results.append(

                metrics

            )



    result = pd.DataFrame(

        all_results

    )


    result = result[

        [

            "Model",

            "Method",

            "Precision",

            "Recall",

            "F1"

        ]

    ]



    print("="*70)

    print(

        "AI4I Anomaly Detection Performance"

    )

    print("="*70)


    print(

        result.to_string(

            index=False

        )

    )



    output = (

        RESULT_DIR

        /

        "AI4I_anomaly_results.csv"

    )


    result.to_csv(

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