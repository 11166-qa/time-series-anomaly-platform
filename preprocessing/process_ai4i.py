from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import joblib



# ======================================================
# Path
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


RAW_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "raw"
    /
    "ai4i2020.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "AI4I"
)



# ======================================================
# Main
# ======================================================

def main():

    print("=" * 70)
    print("Processing AI4I Predictive Maintenance Dataset")
    print("=" * 70)



    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )



    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    df = pd.read_csv(
        RAW_FILE
    )


    print(
        "Original shape:",
        df.shape
    )



    # --------------------------------------------------
    # 2. Create timestamp
    # --------------------------------------------------

    # AI4I没有真实时间序列时间戳
    # 构造分钟级时间索引

    df["date"] = pd.date_range(

        start="2020-01-01",

        periods=len(df),

        freq="min"

    )



    # --------------------------------------------------
    # 3. Save labels
    # --------------------------------------------------

    labels = pd.DataFrame({

        "date":
            df["date"],

        "label":
            df["Machine failure"]

    })


    labels.to_csv(

        OUTPUT_DIR
        /
        "labels.csv",

        index=False

    )



    # --------------------------------------------------
    # 4. Select forecasting features
    # --------------------------------------------------

    feature_columns = [

        "Air temperature [K]",

        "Process temperature [K]",

        "Rotational speed [rpm]",

        "Torque [Nm]",

        "Tool wear [min]"

    ]



    data = df[

        ["date"]

        +

        feature_columns

    ].copy()



    print(
        "Selected features:"
    )

    for c in feature_columns:

        print(
            " -",
            c
        )



    # --------------------------------------------------
    # 5. Split data
    # --------------------------------------------------

    n = len(data)


    train_end = int(
        n * 0.7
    )


    val_end = int(
        n * 0.85
    )


    train = data.iloc[
        :train_end
    ].copy()


    val = data.iloc[
        train_end:val_end
    ].copy()


    test = data.iloc[
        val_end:
    ].copy()



    # --------------------------------------------------
    # 6. Scaling
    # --------------------------------------------------

    scaler = StandardScaler()


    train_values = train[
        feature_columns
    ]


    scaler.fit(
        train_values
    )


    train[
        feature_columns
    ] = scaler.transform(
        train_values
    )


    val[
        feature_columns
    ] = scaler.transform(
        val[feature_columns]
    )


    test[
        feature_columns
    ] = scaler.transform(
        test[feature_columns]
    )



    # --------------------------------------------------
    # 7. Save scaler
    # --------------------------------------------------

    joblib.dump(

        scaler,

        OUTPUT_DIR
        /
        "scaler.joblib"

    )



    # --------------------------------------------------
    # 8. Save csv
    # --------------------------------------------------

    train.to_csv(

        OUTPUT_DIR
        /
        "train.csv",

        index=False

    )


    val.to_csv(

        OUTPUT_DIR
        /
        "val.csv",

        index=False

    )


    test.to_csv(

        OUTPUT_DIR
        /
        "test.csv",

        index=False

    )



    print()

    print(
        "Processed dataset saved:"
    )

    print(
        OUTPUT_DIR
    )


    print()

    print(
        "Train:",
        train.shape
    )

    print(
        "Val:",
        val.shape
    )

    print(
        "Test:",
        test.shape
    )


    print("=" * 70)



if __name__ == "__main__":

    main()