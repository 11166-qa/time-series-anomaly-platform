from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "ETTh1"
    / "distribution_statistics.csv"
)


def main():

    scaler = joblib.load(
        DATA_DIR / "scaler.joblib"
    )

    feature_columns = list(
        scaler.feature_names_in_
    )

    rows = []

    for split in [
        "train",
        "val",
        "test",
    ]:

        df = pd.read_csv(
            DATA_DIR / f"{split}.csv"
        )

        # 恢复原始尺度
        restored = (
            scaler.inverse_transform(
                df[feature_columns]
            )
        )

        restored_df = pd.DataFrame(
            restored,
            columns=feature_columns,
        )

        for feature in feature_columns:

            rows.append({
                "split": split,
                "variable": feature,
                "mean":
                    restored_df[feature].mean(),

                "std":
                    restored_df[feature].std(),

                "min":
                    restored_df[feature].min(),

                "max":
                    restored_df[feature].max(),
            })

    result = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(result.to_string(index=False))

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()