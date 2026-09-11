import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return mean_absolute_error(
        y_true,
        y_pred,
    )


def r2(y_true, y_pred):
    """
    Coefficient of Determination
    """
    return r2_score(
        y_true,
        y_pred,
    )


def mape(
    y_true,
    y_pred,
    epsilon=1e-6,
):
    """
    Mean Absolute Percentage Error.

    为避免真实值接近 0 时分母异常，
    忽略绝对值 <= epsilon 的真实值。
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = (
        np.abs(y_true) > epsilon
    )

    if not np.any(mask):
        return np.nan

    percentage_error = (
        np.abs(
            (
                y_true[mask]
                - y_pred[mask]
            )
            / y_true[mask]
        )
    )

    return (
        np.mean(percentage_error)
        * 100.0
    )


def regression_metrics(
    y_true,
    y_pred,
):
    """
    统一计算回归评价指标。
    """

    return {
        "R2": r2(
            y_true,
            y_pred,
        ),
        "RMSE": rmse(
            y_true,
            y_pred,
        ),
        "MAE": mae(
            y_true,
            y_pred,
        ),
        "MAPE": mape(
            y_true,
            y_pred,
        ),
    }