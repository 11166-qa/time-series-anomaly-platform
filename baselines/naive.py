import torch


def persistence_forecast(x, pred_len):
    """
    Persistence / Last Value Baseline

    x:
        [B, lookback, features]

    output:
        [B, pred_len, features]

    使用输入序列最后一个时间点，
    预测全部未来时间点。
    """

    last_value = x[:, -1:, :]

    prediction = last_value.repeat(
        1,
        pred_len,
        1,
    )

    return prediction


def seasonal_naive_forecast(
    x,
    pred_len,
    season_length=24,
):
    """
    Seasonal Naive Baseline

    对小时级数据默认使用24小时周期。

    例如：
        前一天的 08:00
        ↓
        预测第二天的 08:00
    """

    if x.size(1) < season_length:
        raise ValueError(
            f"lookback={x.size(1)} "
            f"小于 season_length={season_length}"
        )

    seasonal_pattern = x[
        :,
        -season_length:,
        :
    ]

    # 支持 pred_len > season_length 的情况
    repeat_times = (
        pred_len
        + season_length
        - 1
    ) // season_length

    prediction = seasonal_pattern.repeat(
        1,
        repeat_times,
        1,
    )

    prediction = prediction[
        :,
        :pred_len,
        :
    ]

    return prediction