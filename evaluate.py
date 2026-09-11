from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)
from models.lstm import LSTMForecaster
from utils.metrics import regression_metrics


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ETTh1"
    / "LSTM"
    / "baseline"
)

MODEL_PATH = (
    RESULT_DIR
    / "best_model.pt"
)

SCALER_PATH = (
    DATA_DIR
    / "scaler.joblib"
)


# ============================================================
# 2. 配置
# ============================================================

BATCH_SIZE = 64


# ============================================================
# 3. 加载模型
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_config = (
        checkpoint[
            "model_config"
        ]
    )

    # 兼容之前没有保存 dropout 的 checkpoint
    dropout = model_config.get(
        "dropout",
        0.2,
    )

    model = LSTMForecaster(
        input_size=
            model_config["input_size"],

        hidden_size=
            model_config["hidden_size"],

        num_layers=
            model_config["num_layers"],

        pred_len=
            model_config["pred_len"],

        output_size=
            model_config["output_size"],

        dropout=
            dropout,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)

    model.eval()

    return model, checkpoint


# ============================================================
# 4. 模型预测
# ============================================================

def predict(
    model,
    dataloader,
    device,
):

    predictions = []
    targets = []

    with torch.inference_mode():

        for x, y in dataloader:

            x = x.to(device)

            prediction = model(x)

            predictions.append(
                prediction.cpu().numpy()
            )

            targets.append(
                y.numpy()
            )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    targets = np.concatenate(
        targets,
        axis=0,
    )

    return predictions, targets


# ============================================================
# 5. 反标准化
# ============================================================

def inverse_transform_3d(
    data,
    scaler,
):
    """
    [samples, pred_len, features]
        ↓
    [samples * pred_len, features]
        ↓
    inverse_transform
        ↓
    恢复3维
    """

    original_shape = data.shape

    reshaped = data.reshape(
        -1,
        original_shape[-1],
    )

    restored = scaler.inverse_transform(
        reshaped
    )

    restored = restored.reshape(
        original_shape
    )

    return restored


# ============================================================
# 6. 计算每个变量的指标
# ============================================================

def calculate_variable_metrics(
    targets,
    predictions,
    feature_columns,
):

    rows = []

    for feature_index, feature_name in enumerate(
        feature_columns
    ):

        y_true = (
            targets[
                :,
                :,
                feature_index
            ]
            .reshape(-1)
        )

        y_pred = (
            predictions[
                :,
                :,
                feature_index
            ]
            .reshape(-1)
        )

        metrics = regression_metrics(
            y_true,
            y_pred,
        )

        rows.append({
            "variable":
                feature_name,

            **metrics,
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. Macro 指标
# ============================================================

def calculate_macro_metrics(
    variable_metrics,
):

    metric_columns = [
        "R2",
        "RMSE",
        "MAE",
        "MAPE",
    ]

    macro = (
        variable_metrics[
            metric_columns
        ]
        .mean()
    )

    return pd.DataFrame([
        {
            "metric_type":
                "macro_average",

            "R2":
                macro["R2"],

            "RMSE":
                macro["RMSE"],

            "MAE":
                macro["MAE"],

            "MAPE":
                macro["MAPE"],
        }
    ])


# ============================================================
# 8. Horizon 指标
# ============================================================

def calculate_horizon_metrics(
    targets,
    predictions,
):
    """
    分析未来第 1~pred_len 个时间点的预测性能。

    每个 horizon 将全部变量展开后计算。
    """

    pred_len = (
        targets.shape[1]
    )

    rows = []

    for horizon in range(pred_len):

        y_true = (
            targets[
                :,
                horizon,
                :
            ]
            .reshape(-1)
        )

        y_pred = (
            predictions[
                :,
                horizon,
                :
            ]
            .reshape(-1)
        )

        metrics = regression_metrics(
            y_true,
            y_pred,
        )

        rows.append({
            "horizon":
                horizon + 1,

            **metrics,
        })

    return pd.DataFrame(rows)


# ============================================================
# 9. 聚合重叠预测
# ============================================================

def build_aggregated_predictions(
    dataset,
    targets,
    predictions,
):
    """
    多步滑动预测时，同一个真实时间点可能被多个预测窗口预测。

    例如某时刻可能同时对应：
        前一个窗口的 horizon=2
        另一个窗口的 horizon=1

    为后续可视化和异常检测，
    对同一 timestamp、variable 的多个预测取平均。
    """

    records = []

    num_samples = (
        predictions.shape[0]
    )

    pred_len = (
        predictions.shape[1]
    )

    num_features = (
        predictions.shape[2]
    )

    for sample_index in range(
        num_samples
    ):

        timestamps = (
            dataset
            .get_target_timestamps(
                sample_index
            )
        )

        for horizon in range(
            pred_len
        ):

            timestamp = (
                timestamps[horizon]
            )

            for feature_index in range(
                num_features
            ):

                records.append({
                    "timestamp":
                        timestamp,

                    "variable":
                        dataset.feature_columns[
                            feature_index
                        ],

                    "actual":
                        targets[
                            sample_index,
                            horizon,
                            feature_index,
                        ],

                    "prediction":
                        predictions[
                            sample_index,
                            horizon,
                            feature_index,
                        ],
                })

    df = pd.DataFrame(
        records
    )

    aggregated = (
        df
        .groupby(
            [
                "timestamp",
                "variable",
            ],
            as_index=False,
        )
        .agg({
            "actual":
                "mean",

            "prediction":
                "mean",
        })
    )

    aggregated[
        "residual"
    ] = np.abs(
        aggregated["actual"]
        - aggregated["prediction"]
    )

    return aggregated


# ============================================================
# 10. Main
# ============================================================

def main():

    print("=" * 70)
    print("ETTh1 LSTM Evaluation")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # 检查文件
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"找不到Scaler：{SCALER_PATH}"
        )

    # --------------------------------------------------------
    # 加载 Checkpoint
    # --------------------------------------------------------

    model, checkpoint = load_model(
        MODEL_PATH,
        device,
    )

    task_config = (
        checkpoint[
            "task_config"
        ]
    )

    lookback = (
        task_config[
            "lookback"
        ]
    )

    pred_len = (
        task_config[
            "pred_len"
        ]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['val_loss']:.6f}"
    )

    print(
        f"Lookback: {lookback}"
    )

    print(
        f"Prediction length: "
        f"{pred_len}"
    )

    # --------------------------------------------------------
    # Test Dataset
    #
    # Test 的历史上下文来自 Validation 的最后 lookback 个时间点
    # --------------------------------------------------------

    test_dataset = TimeSeriesDataset(
        csv_path=
            DATA_DIR / "test.csv",

        context_csv_path=
            DATA_DIR / "val.csv",

        lookback=
            lookback,

        pred_len=
            pred_len,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    print(
        f"Features: "
        f"{test_dataset.feature_columns}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions_scaled, targets_scaled = (
        predict(
            model=model,
            dataloader=test_loader,
            device=device,
        )
    )

    print(
        f"\nPrediction shape: "
        f"{predictions_scaled.shape}"
    )

    print(
        f"Target shape: "
        f"{targets_scaled.shape}"
    )

    # --------------------------------------------------------
    # 加载 Scaler
    # --------------------------------------------------------

    scaler = joblib.load(
        SCALER_PATH
    )

    # --------------------------------------------------------
    # 反标准化
    # --------------------------------------------------------

    predictions = inverse_transform_3d(
        predictions_scaled,
        scaler,
    )

    targets = inverse_transform_3d(
        targets_scaled,
        scaler,
    )

    # --------------------------------------------------------
    # 每变量指标
    # --------------------------------------------------------

    variable_metrics = (
        calculate_variable_metrics(
            targets=targets,
            predictions=predictions,
            feature_columns=
                test_dataset.feature_columns,
        )
    )

    # --------------------------------------------------------
    # Macro指标
    # --------------------------------------------------------

    overall_metrics = (
        calculate_macro_metrics(
            variable_metrics
        )
    )

    # --------------------------------------------------------
    # Horizon指标
    # --------------------------------------------------------

    horizon_metrics = (
        calculate_horizon_metrics(
            targets=targets,
            predictions=predictions,
        )
    )

    # --------------------------------------------------------
    # 聚合预测
    # --------------------------------------------------------

    aggregated_predictions = (
        build_aggregated_predictions(
            dataset=test_dataset,
            targets=targets,
            predictions=predictions,
        )
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    variable_metrics.to_csv(
        RESULT_DIR
        / "variable_metrics.csv",
        index=False,
    )

    overall_metrics.to_csv(
        RESULT_DIR
        / "overall_metrics.csv",
        index=False,
    )

    horizon_metrics.to_csv(
        RESULT_DIR
        / "horizon_metrics.csv",
        index=False,
    )

    aggregated_predictions.to_csv(
        RESULT_DIR
        / "predictions_aggregated.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 打印
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Variable Metrics")
    print("=" * 70)

    print(
        variable_metrics.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("Macro Average")
    print("=" * 70)

    print(
        overall_metrics.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("Evaluation Finished")
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{RESULT_DIR}"
    )


if __name__ == "__main__":
    main()