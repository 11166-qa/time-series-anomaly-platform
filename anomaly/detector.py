import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest



def detect_by_3sigma(
        df,
        column="residual"
):

    mean = df[column].mean()

    std = df[column].std()


    threshold = mean + 3 * std


    result = df.copy()


    result["anomaly_3sigma"] = (
        result[column] > threshold
    ).astype(int)


    return result, threshold





def detect_by_iqr(
        df,
        column="residual"
):

    q1 = df[column].quantile(0.25)

    q3 = df[column].quantile(0.75)


    iqr = q3 - q1


    threshold = q3 + 1.5 * iqr


    result = df.copy()


    result["anomaly_iqr"] = (
        result[column] > threshold
    ).astype(int)


    return result, threshold





def detect_by_isolation_forest(
        df,
        column="residual",
        contamination=0.01
):


    model = IsolationForest(

        contamination=contamination,

        random_state=42

    )


    x = df[[column]]


    prediction = model.fit_predict(
        x
    )


    result = df.copy()


    # sklearn:
    # 1 normal
    # -1 anomaly

    result["anomaly_iforest"] = (

        prediction == -1

    ).astype(int)



    return result





def combine_detection(
        df
):


    result, sigma_threshold = detect_by_3sigma(
        df
    )


    result, iqr_threshold = detect_by_iqr(
        result
    )


    result = detect_by_isolation_forest(
        result
    )


    return result, {

        "3sigma_threshold":
            float(sigma_threshold),

        "iqr_threshold":
            float(iqr_threshold),

        "samples":
            int(len(result)),

        "3sigma_anomaly_count":
            int(
                result["anomaly_3sigma"].sum()
            ),

        "iqr_anomaly_count":
            int(
                result["anomaly_iqr"].sum()
            ),

        "iforest_anomaly_count":
            int(
                result["anomaly_iforest"].sum()
            ),

    }