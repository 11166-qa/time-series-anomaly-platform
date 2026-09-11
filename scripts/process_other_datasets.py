from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler



PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


RAW_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "raw"
)


PROCESSED_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
)



DATASETS = {

    "Electricity":
        "electricity.csv",

    "Weather":
        "weather.csv",

}



def process_dataset(
        name,
        filename,
):


    print(
        "\n"
        +
        "=" * 70
    )


    print(
        f"Processing {name}"
    )


    print(
        "=" * 70
    )


    input_path = (
        RAW_DIR
        /
        filename
    )


    output_dir = (
        PROCESSED_DIR
        /
        name
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )



    # ==========================
    # read csv
    # ==========================


    df = pd.read_csv(
        input_path
    )


    print(
        "Original shape:",
        df.shape
    )



    # ==========================
    # sort by time
    # ==========================


    df["date"] = pd.to_datetime(
        df["date"]
    )


    df = (
        df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )



    # ==========================
    # split
    # 70 / 10 / 20
    # ==========================


    n = len(df)


    train_end = int(
        n * 0.7
    )


    val_end = int(
        n * 0.8
    )


    train_df = (
        df.iloc[
            :train_end
        ]
        .copy()
    )


    val_df = (
        df.iloc[
            train_end:val_end
        ]
        .copy()
    )


    test_df = (
        df.iloc[
            val_end:
        ]
        .copy()
    )



    print(
        "Train:",
        train_df.shape
    )


    print(
        "Val:",
        val_df.shape
    )


    print(
        "Test:",
        test_df.shape
    )



    # ==========================
    # scaler
    # ==========================


    feature_columns = [
        c
        for c in df.columns
        if c != "date"
    ]



    scaler = StandardScaler()



    scaler.fit(
        train_df[
            feature_columns
        ]
    )



    train_df[
        feature_columns
    ] = scaler.transform(
        train_df[
            feature_columns
        ]
    )


    val_df[
        feature_columns
    ] = scaler.transform(
        val_df[
            feature_columns
        ]
    )


    test_df[
        feature_columns
    ] = scaler.transform(
        test_df[
            feature_columns
        ]
    )



    # ==========================
    # save
    # ==========================


    train_df.to_csv(
        output_dir
        /
        "train.csv",
        index=False,
    )


    val_df.to_csv(
        output_dir
        /
        "val.csv",
        index=False,
    )


    test_df.to_csv(
        output_dir
        /
        "test.csv",
        index=False,
    )


    joblib.dump(

        scaler,

        output_dir
        /
        "scaler.joblib"

    )



    print(
        f"{name} finished."
    )



def main():

    for name, filename in DATASETS.items():

        process_dataset(
            name,
            filename,
        )



if __name__ == "__main__":

    main()