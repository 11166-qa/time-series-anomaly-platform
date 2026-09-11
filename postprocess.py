from pathlib import Path

import pandas as pd



def merge_predictions(
        input_file,
        output_file
):

    df = pd.read_csv(
        input_file
    )


    print(
        "Original samples:",
        len(df)
    )


    # 同一时间多个预测取平均
    merged = (

        df

        .groupby(
            "date",
            as_index=False
        )

        .agg({

            "target":
                "first",

            "prediction":
                "mean",

        })

    )


    # 重新计算残差

    merged["residual"] = (

        merged["target"]

        -

        merged["prediction"]

    ).abs()



    merged.to_csv(

        output_file,

        index=False,

    )


    print(
        "Merged samples:",
        len(merged)
    )


    print(
        "Saved:",
        output_file
    )




if __name__ == "__main__":


    result_dir = Path(

        "results/Weather/iTransformer/product"

    )


    merge_predictions(

        result_dir / "predictions.csv",

        result_dir / "predictions_merged.csv"

    )