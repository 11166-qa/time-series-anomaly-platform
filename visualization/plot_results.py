from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# ==========================================================
# 时间轴设置
# ==========================================================

def set_time_ticks(
        ax,
        data,
        num_ticks=8
):

    length = len(data)

    positions = np.linspace(
        0,
        length - 1,
        num_ticks,
        dtype=int
    )


    labels = data["date"].iloc[
        positions
    ]


    ax.set_xticks(
        positions
    )


    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=9
    )



# ==========================================================
# 预测结果可视化
# ==========================================================

def plot_prediction(
        df,
        save_path,
        sample_num=1000
):

    data = (
        df
        .iloc[:sample_num]
        .reset_index(drop=True)
    )


    fig, ax = plt.subplots(
        figsize=(12,4)
    )


    ax.plot(

        data.index,

        data["target"],

        linewidth=1.5,

        label="True"

    )


    ax.plot(

        data.index,

        data["prediction"],

        linewidth=1.5,

        label="Prediction"

    )


    ax.set_xlabel(
        "Time"
    )


    ax.set_ylabel(
        "Temperature"
    )


    ax.set_title(
        "Prediction vs True"
    )


    set_time_ticks(
        ax,
        data
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()





# ==========================================================
# 残差可视化
# ==========================================================

def plot_residual(

        df,

        save_path,

        sample_num=1000

):


    data = (
        df
        .iloc[:sample_num]
        .reset_index(drop=True)
    )


    residual = data["residual"]


    threshold = (

        residual.mean()

        +

        3 * residual.std()

    )



    anomaly = data[

        data["residual"]

        >

        threshold

    ]



    fig, ax = plt.subplots(

        figsize=(12,4)

    )



    ax.plot(

        data.index,

        residual,

        linewidth=1.5,

        label="Residual"

    )


    # 3sigma阈值线

    ax.axhline(

        threshold,

        linestyle="--",

        linewidth=1.5,

        label=f"3σ threshold ({threshold:.2f})"

    )


    # 异常点

    ax.scatter(

        anomaly.index,

        anomaly["residual"],

        s=45,

        color="red",

        edgecolors="black",

        linewidths=0.5,

        label="Anomaly"

    )



    ax.set_xlabel(

        "Time"

    )


    ax.set_ylabel(

        "Residual"

    )


    ax.set_title(

        "Prediction Residual"

    )


    set_time_ticks(

        ax,

        data

    )


    ax.legend()



    plt.tight_layout()


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()





# ==========================================================
# 异常检测可视化
# ==========================================================

def plot_anomaly(

        df,

        save_path,

        sample_num=1000

):


    data = (

        df

        .iloc[:sample_num]

        .reset_index(drop=True)

    )


    anomaly = data[

        data["anomaly_3sigma"]

        ==

        1

    ]



    fig, ax = plt.subplots(

        figsize=(12,4)

    )



    ax.plot(

        data.index,

        data["target"],

        linewidth=1.5,

        label="Target"

    )



    ax.scatter(

        anomaly.index,

        anomaly["target"],

        s=55,

        color="red",

        edgecolors="black",

        linewidths=0.5,

        label="Anomaly"

    )



    ax.set_xlabel(

        "Time"

    )


    ax.set_ylabel(

        "Temperature"

    )


    ax.set_title(

        f"Detected Anomalies (3σ): {len(anomaly)} points"

    )


    set_time_ticks(

        ax,

        data

    )


    ax.legend()



    plt.tight_layout()


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()





# ==========================================================
# Main
# ==========================================================

def main():


    result_dir = Path(

        "results/Weather/iTransformer/product"

    )


    output_dir = (

        result_dir

        /

        "figures"

    )


    output_dir.mkdir(

        exist_ok=True

    )



    prediction_df = pd.read_csv(

        result_dir

        /

        "predictions_merged.csv"

    )


    anomaly_df = pd.read_csv(

        result_dir

        /

        "anomaly_results.csv"

    )



    plot_prediction(

        prediction_df,

        output_dir

        /

        "prediction_vs_true.png"

    )


    plot_residual(

        prediction_df,

        output_dir

        /

        "residual_curve.png"

    )


    plot_anomaly(

        anomaly_df,

        output_dir

        /

        "anomaly_detection.png"

    )



    print(
        "Visualization finished."
    )


    print(
        "Saved:",
        output_dir
    )





if __name__ == "__main__":

    main()