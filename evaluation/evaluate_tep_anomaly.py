from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.ensemble import IsolationForest



# ======================================================
# Path
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


LABEL_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "TEP"
    /
    "labels.csv"
)


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


# 当前 TEP 配置
LOOKBACK = 48
PRED_LEN = 6





# ======================================================
# Detection methods
# ======================================================

def detect_3sigma(residual):

    threshold = (

        residual.mean()

        +

        3 * residual.std()

    )

    return (

        residual > threshold

    ).astype(int)





def detect_iqr(residual):

    q1 = residual.quantile(
        0.25
    )

    q3 = residual.quantile(
        0.75
    )


    iqr = q3 - q1


    threshold = (

        q3

        +

        1.5 * iqr

    )


    return (

        residual > threshold

    ).astype(int)





def detect_isolation_forest(residual):

    model = IsolationForest(

        contamination=0.1,

        random_state=42

    )


    pred = model.fit_predict(

        residual.values.reshape(-1, 1)

    )


    return (

        pred == -1

    ).astype(int)





# ======================================================
# Metrics
# ======================================================

def calculate_metrics(y_true, y_pred):

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





# ======================================================
# Align labels and residuals
# ======================================================

def align_labels(residual_df, labels):

    """
    TEP没有真实时间戳，
    按预测样本顺序对齐标签。

    predict.py由于滑动窗口:
    前 lookback + pred_len - 1 个点无法产生预测
    """

    residual_len = len(residual_df)


    labels_len = len(labels)



    # 预测开始位置

    offset = LOOKBACK + PRED_LEN - 1



    labels = labels.iloc[

        offset:

    ].reset_index(

        drop=True

    )



    # 长度匹配

    min_len = min(

        residual_len,

        len(labels)

    )



    aligned_labels = labels.iloc[

        :min_len

    ]


    residual_df = residual_df.iloc[

        :min_len

    ].copy()



    residual_df["label"] = (

        aligned_labels["label"]

        .values

    )


    return residual_df





# ======================================================
# Main
# ======================================================

def main():


    print("=" * 70)

    print(
        "TEP Anomaly Detection Performance"
    )

    print("=" * 70)



    labels = pd.read_csv(

        LABEL_FILE

    )



    results = []



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



        df = align_labels(

            residual_df,

            labels

        )



        print(

            model,

            "matched samples:",

            len(df)

        )



        if len(df) == 0:

            continue



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

                detect_isolation_forest(

                    df["residual"]

                )

        }



        for method, pred in methods.items():


            metrics = calculate_metrics(

                y_true,

                pred

            )


            metrics.update({

                "Model":

                    model,


                "Method":

                    method

            })


            results.append(

                metrics

            )



    if len(results) == 0:

        print(
            "No results generated."
        )

        return



    result_df = pd.DataFrame(

        results

    )


    result_df = result_df[

        [

            "Model",

            "Method",

            "Precision",

            "Recall",

            "F1"

        ]

    ]



    print()

    print(

        result_df.to_string(

            index=False

        )

    )



    output_file = (

        RESULT_DIR

        /

        "TEP_anomaly_results.csv"

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