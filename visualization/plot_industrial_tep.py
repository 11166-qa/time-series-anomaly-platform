from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



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


FIG_DIR = (

    RESULT_DIR
    /
    "figures"

)


FIG_DIR.mkdir(

    parents=True,

    exist_ok=True

)



# ======================================================
# 1. Prediction visualization
# ======================================================

def plot_prediction():


    file = (

        RESULT_DIR
        /
        "forecasting"
        /
        "iTransformer"
        /
        "predictions.csv"

    )


    df = pd.read_csv(file)



    # 取前1000个点，保证展示清晰

    df = df.iloc[:1000]



    plt.figure(

        figsize=(10,4)

    )


    plt.plot(

        df["sample_index"],

        df["target"],

        label="True"

    )


    plt.plot(

        df["sample_index"],

        df["prediction"],

        label="Prediction"

    )


    plt.xlabel(

        "Sample index"

    )


    plt.ylabel(

        "Process value"

    )


    plt.title(

        "TEP Normal Process Forecasting (iTransformer)"

    )


    plt.legend()


    plt.tight_layout()



    plt.savefig(

        FIG_DIR
        /
        "TEP_prediction_iTransformer.png",

        dpi=300

    )


    plt.close()





# ======================================================
# 2. Residual visualization
# ======================================================

def plot_residual():


    file = (

        RESULT_DIR
        /
        "anomaly"
        /
        "iTransformer"
        /
        "residuals.csv"

    )


    df = pd.read_csv(file)



    df = df.iloc[:1500]



    plt.figure(

        figsize=(10,4)

    )


    plt.plot(

        df["sample_index"],

        df["residual"]

    )


    plt.xlabel(

        "Sample index"

    )


    plt.ylabel(

        "Residual"

    )


    plt.title(

        "TEP Fault Residual Analysis (iTransformer)"

    )


    plt.tight_layout()



    plt.savefig(

        FIG_DIR
        /
        "TEP_fault_residual.png",

        dpi=300

    )


    plt.close()





# ======================================================
# 3. Anomaly detection visualization
# ======================================================

def plot_anomaly():


    residual_file = (

        RESULT_DIR
        /
        "anomaly"
        /
        "iTransformer"
        /
        "residuals.csv"

    )


    label_file = (

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


    residual = pd.read_csv(

        residual_file

    )


    labels = pd.read_csv(

        label_file

    )


    # 对齐长度

    offset = 48 + 6 - 1


    labels = labels.iloc[offset:]



    n = min(

        len(residual),

        len(labels)

    )


    residual = residual.iloc[:n]

    labels = labels.iloc[:n]



    # IQR异常检测

    q1 = residual["residual"].quantile(0.25)

    q3 = residual["residual"].quantile(0.75)


    threshold = q3 + 1.5*(q3-q1)


    anomaly = (

        residual["residual"]

        >

        threshold

    )



    plt.figure(

        figsize=(10,4)

    )


    plt.plot(

        residual["sample_index"],

        residual["residual"],

        label="Residual"

    )


    plt.scatter(

        residual.loc[

            anomaly,

            "sample_index"

        ],

        residual.loc[

            anomaly,

            "residual"

        ],

        label="Detected anomaly"

    )


    plt.xlabel(

        "Sample index"

    )


    plt.ylabel(

        "Residual"

    )


    plt.title(

        "TEP Fault Detection Result (iTransformer + IQR)"

    )


    plt.legend()


    plt.tight_layout()



    plt.savefig(

        FIG_DIR
        /
        "TEP_anomaly_detection.png",

        dpi=300

    )


    plt.close()





# ======================================================
# Main
# ======================================================

def main():


    print(

        "Generating TEP industrial figures..."

    )


    plot_prediction()


    plot_residual()


    plot_anomaly()



    print(

        "Saved:",

        FIG_DIR

    )





if __name__ == "__main__":

    main()