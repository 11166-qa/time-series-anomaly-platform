from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ETTh1"
    / "LSTM"
    / "baseline"
)

FIGURE_DIR = (
    RESULT_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def plot_training_history():

    history = pd.read_csv(
        RESULT_DIR
        / "training_history.csv"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("ETTh1 LSTM Training History")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "training_history.png",
        dpi=300,
    )

    plt.close()


def plot_prediction(variable, max_points=500):

    df = pd.read_csv(
        RESULT_DIR
        / "predictions_aggregated.csv"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    data = (
        df[df["variable"] == variable]
        .sort_values("timestamp")
        .head(max_points)
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        data["timestamp"],
        data["actual"],
        label="Actual",
    )

    plt.plot(
        data["timestamp"],
        data["prediction"],
        label="LSTM",
    )

    plt.xlabel("Time")
    plt.ylabel(variable)

    plt.title(
        f"ETTh1 LSTM Prediction - {variable}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / f"prediction_{variable}.png",
        dpi=300,
    )

    plt.close()


def plot_horizon_metrics():

    df = pd.read_csv(
        RESULT_DIR
        / "horizon_metrics.csv"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["horizon"],
        df["R2"],
        marker="o",
    )

    plt.xlabel("Forecast Horizon (hour)")
    plt.ylabel("R²")
    plt.title(
        "ETTh1 LSTM Forecast Horizon Performance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "horizon_r2.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["horizon"],
        df["RMSE"],
        marker="o",
    )

    plt.xlabel("Forecast Horizon (hour)")
    plt.ylabel("RMSE")

    plt.title(
        "ETTh1 LSTM RMSE vs Forecast Horizon"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "horizon_rmse.png",
        dpi=300,
    )

    plt.close()


def main():

    plot_training_history()

    # 三个有代表性的变量
    plot_prediction("OT")
    plot_prediction("MUFL")
    plot_prediction("LUFL")

    plot_horizon_metrics()

    print(
        f"Figures saved to:\n{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()