from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
import joblib



# ======================================================
# Path
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


RAW_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "raw"
    /
    "TEP"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "TEP"
)



NORMAL_FILE = (
    RAW_DIR
    /
    "mode1_normal_500.xlsx"
)



FAULT_FILES = [

    "mode1_10_1.xlsx",

    "mode1_11_1.xlsx",

    "mode1_12_1.xlsx",

    "mode1_13_1.xlsx",

    "mode1_14_1.xlsx",

]



# ======================================================
# Features
# ======================================================

FEATURES = [

    f"feature_{i}"

    for i in range(1, 42)

]





# ======================================================
# Load normal data
# ======================================================

def load_normal():

    df = pd.read_excel(
        NORMAL_FILE
    )


    # Time -> date

    df = df.rename(

        columns={

            "Time":

            "date"

        }

    )


    # xmv -> feature

    rename_dict = {

        f"xmv-{i}":

        f"feature_{i}"

        for i in range(1,42)

    }


    df = df.rename(

        columns=rename_dict

    )


    df = df[

        [

            "date"

        ]

        +

        FEATURES

    ]


    # 字符串采样索引

    df["date"] = (

        df["date"]

        .index

        .astype(str)

    )


    df["label"] = 0


    return df





# ======================================================
# Load fault data
# ======================================================

def load_fault():

    fault_list = []



    for file in FAULT_FILES:


        path = (

            RAW_DIR

            /

            "faults"

            /

            file

        )


        print(

            "Loading:",

            file

        )


        df = pd.read_excel(

            path

        )



        df = df.rename(

            columns={

                "Time (h)":

                "date"

            }

        )



        # XMEAS -> feature

        rename_dict = {

            f"XMEAS-{i}":

            f"feature_{i}"

            for i in range(1,42)

        }



        df = df.rename(

            columns=rename_dict

        )



        df = df[

            [

                "date"

            ]

            +

            FEATURES

        ]



        # 字符串采样索引

        df["date"] = (

            df["date"]

            .index

            .astype(str)

        )


        df["label"] = 1



        fault_list.append(

            df

        )



    return pd.concat(

        fault_list,

        ignore_index=True

    )





# ======================================================
# Main
# ======================================================

def main():

    print("="*70)

    print(

        "Processing TEP Dataset"

    )

    print("="*70)



    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    # Load

    normal = load_normal()

    fault = load_fault()



    print()

    print(

        "Normal:",

        normal.shape

    )


    print(

        "Fault:",

        fault.shape

    )



    # Split normal

    n = len(normal)


    train_end = int(

        n*0.7

    )


    val_end = int(

        n*0.85

    )



    train = normal.iloc[

        :train_end

    ].copy()



    val = normal.iloc[

        train_end:val_end

    ].copy()



    normal_test = normal.iloc[

        val_end:

    ].copy()



    test = fault.copy()



    # Scaling

    scaler = StandardScaler()



    scaler.fit(

        train[FEATURES]

    )



    for df in [

        train,

        val,

        normal_test,

        test

    ]:


        df[FEATURES] = scaler.transform(

            df[FEATURES]

        )



    # Save scaler

    joblib.dump(

        scaler,

        OUTPUT_DIR

        /

        "scaler.joblib"

    )



    columns = [

        "date"

    ] + FEATURES



    # Save data

    train[columns].to_csv(

        OUTPUT_DIR

        /

        "train.csv",

        index=False

    )


    val[columns].to_csv(

        OUTPUT_DIR

        /

        "val.csv",

        index=False

    )


    normal_test[columns].to_csv(

        OUTPUT_DIR

        /

        "normal_test.csv",

        index=False

    )


    test[columns].to_csv(

        OUTPUT_DIR

        /

        "test.csv",

        index=False

    )



    labels = test[

        [

            "date",

            "label"

        ]

    ]


    labels.to_csv(

        OUTPUT_DIR

        /

        "labels.csv",

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

        "Normal Test:",

        normal_test.shape

    )


    print(

        "Fault Test:",

        test.shape

    )


    print("="*70)





if __name__ == "__main__":

    main()




if __name__ == "__main__":

    main()