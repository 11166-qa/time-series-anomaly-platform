from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt



PROJECT_ROOT = Path(__file__).resolve().parent.parent


RESULT_DIR = (
    PROJECT_ROOT
    /
    "results"
    /
    "AI4I"
)


DATA_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "AI4I"
)


FIG_DIR = RESULT_DIR / "figures"

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ======================================================
# 1. Sensor trends
# ======================================================

def plot_sensor_trends():


    df = pd.read_csv(

        DATA_DIR
        /
        "test.csv"

    )


    plt.figure(
        figsize=(10,5)
    )


    sensors = [

        "Air temperature [K]",

        "Process temperature [K]",

        "Rotational speed [rpm]",

        "Torque [Nm]"

    ]


    for col in sensors:

        if col in df.columns:

            plt.plot(

                df[col].values[:1000],

                label=col

            )


    plt.xlabel(
        "Sample index"
    )


    plt.ylabel(
        "Value"
    )


    plt.title(
        "AI4I Industrial Sensor Trends"
    )


    plt.legend(
        fontsize=8
    )


    plt.tight_layout()


    plt.savefig(

        FIG_DIR
        /
        "AI4I_sensor_trends.png",

        dpi=300

    )


    plt.close()





# ======================================================
# 2. Prediction
# ======================================================

def plot_prediction():


    file = (

        RESULT_DIR
        /
        "LSTM"
        /
        "case_study"
        /
        "predictions.csv"

    )


    df = pd.read_csv(file)


    df = df.iloc[:1000]


    plt.figure(
        figsize=(10,4)
    )


    plt.plot(

        df["target"],

        label="True"

    )


    plt.plot(

        df["prediction"],

        label="Prediction"

    )


    plt.xlabel(
        "Sample index"
    )


    plt.ylabel(
        "Value"
    )


    plt.title(
        "AI4I Equipment State Forecasting (LSTM)"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        FIG_DIR
        /
        "AI4I_prediction_LSTM.png",

        dpi=300

    )


    plt.close()





# ======================================================
# 3. Failure detection
# ======================================================

def plot_failure_detection():


    residual_file = (

        RESULT_DIR
        /
        "LSTM"
        /
        "case_study"
        /
        "residuals.csv"

    )


    label_file = (

        DATA_DIR
        /
        "labels.csv"

    )


    residual = pd.read_csv(
        residual_file
    )


    labels = pd.read_csv(
        label_file
    )


    n = min(

        len(residual),

        len(labels)

    )


    residual = residual.iloc[:n]

    labels = labels.iloc[:n]



    threshold = (

        residual["residual"].mean()

        +

        3 *
        residual["residual"].std()

    )


    anomaly = (

        residual["residual"]

        >

        threshold

    )


    plt.figure(
        figsize=(10,4)
    )


    plt.plot(

        residual["residual"],

        label="Residual"

    )


    plt.scatter(

        residual.index[anomaly],

        residual.loc[anomaly,"residual"],

        label="Detected anomaly"

    )


    plt.xlabel(
        "Sample index"
    )


    plt.ylabel(
        "Residual"
    )


    plt.title(
        "AI4I Failure Detection (LSTM + 3sigma)"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        FIG_DIR
        /
        "AI4I_failure_detection.png",

        dpi=300

    )


    plt.close()





def main():


    print(
        "Generating AI4I industrial figures..."
    )


    plot_sensor_trends()

    plot_prediction()

    plot_failure_detection()


    print(
        "Saved:",
        FIG_DIR
    )





if __name__ == "__main__":

    main()