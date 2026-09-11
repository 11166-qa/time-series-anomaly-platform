from pathlib import Path
import json

import pandas as pd


from anomaly.detector import combine_detection





def main():


    result_dir = Path(

        "results/Weather/iTransformer/product"

    )


    input_file = (

        result_dir

        /

        "predictions_merged.csv"

    )


    df = pd.read_csv(
        input_file
    )


    result, summary = combine_detection(
        df
    )


    output_csv = (

        result_dir

        /

        "anomaly_results.csv"

    )


    result.to_csv(
        output_csv,
        index=False
    )


    with open(

        result_dir

        /

        "anomaly_summary.json",

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            summary,

            f,

            indent=4,

            ensure_ascii=False

        )


    print("="*60)

    print("Anomaly Detection Finished")

    print("="*60)


    for k,v in summary.items():

        print(
            f"{k}: {v}"
        )


    print()

    print(
        "Saved:",
        output_csv
    )




if __name__ == "__main__":

    main()