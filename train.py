from pathlib import Path
import argparse
import yaml


from trainers.lstm_trainer import train_lstm
from trainers.transformer_trainer import train_transformer
from trainers.itransformer_trainer import train_itransformer



def load_config(config_path):

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)



def main():

    # =====================================================
    # Argument
    # =====================================================

    parser = argparse.ArgumentParser(
        description="Multivariate Time Series Forecasting"
    )


    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config"
    )


    args = parser.parse_args()



    # =====================================================
    # Load config
    # =====================================================

    config = load_config(
        args.config
    )



    dataset_name = (
        config["dataset"]["name"]
    )


    model_name = (
        config["model"]["name"]
    )


    data_dir = Path(
        config["dataset"]["data_dir"]
    )


    result_dir = Path(
        config["output"]["result_dir"]
    )
    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
            result_dir / "config.yaml",
            "w",
            encoding="utf-8"
    ) as f:

        yaml.dump(
            config,
            f,
            allow_unicode=True
        )



    print("=" * 70)

    print(
        "Multivariate Time Series Forecasting"
    )

    print("=" * 70)


    print(
        f"Dataset : {dataset_name}"
    )


    print(
        f"Model   : {model_name}"
    )


    print(
        f"Data    : {data_dir}"
    )


    print(
        f"Output  : {result_dir}"
    )


    print("=" * 70)



    # =====================================================
    # Train config merge
    # =====================================================

    train_config = {}

    train_config.update(
        config["model"]
    )

    train_config.update(
        config["training"]
    )



    # =====================================================
    # Select trainer
    # =====================================================


    if model_name == "LSTM":


        result = train_lstm(

            data_dir=data_dir,

            result_dir=result_dir,

            config=train_config,

        )


    elif model_name == "Transformer":


        result = train_transformer(

            data_dir=data_dir,

            result_dir=result_dir,

            config=train_config,

        )


    elif model_name == "iTransformer":


        result = train_itransformer(

            data_dir=data_dir,

            result_dir=result_dir,

            config=train_config,

        )


    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )



    # =====================================================
    # Result
    # =====================================================

    print()

    print("=" * 70)

    print(
        "Training Finished"
    )

    print("=" * 70)


    for k, v in result.items():

        print(
            f"{k}: {v}"
        )


    print("=" * 70)




if __name__ == "__main__":

    main()