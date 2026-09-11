from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)


from models.lstm import LSTMForecaster


from utils.early_stopping import EarlyStopping
from utils.seed import set_seed



def train_lstm(
        data_dir,
        result_dir,
        config,
):

    # ========================================================
    # Seed
    # ========================================================

    set_seed(
        config["seed"]
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Device: {device}"
    )



    # ========================================================
    # Dataset
    # ========================================================


    train_dataset = TimeSeriesDataset(

        csv_path=
            data_dir / "train.csv",

        lookback=
            config["lookback"],

        pred_len=
            config["pred_len"],

    )



    val_dataset = TimeSeriesDataset(

        csv_path=
            data_dir / "val.csv",

        context_csv_path=
            data_dir / "train.csv",

        lookback=
            config["lookback"],

        pred_len=
            config["pred_len"],

    )



    # ========================================================
    # DataLoader
    # ========================================================


    train_loader = create_dataloader(

        dataset=train_dataset,

        batch_size=
            config["batch_size"],

        shuffle=True,

        num_workers=0,

    )



    val_loader = create_dataloader(

        dataset=val_dataset,

        batch_size=
            config["batch_size"],

        shuffle=False,

        num_workers=0,

    )



    # ========================================================
    # Model
    # ========================================================


    model = LSTMForecaster(

        input_size=
            train_dataset.num_features,

        hidden_size=
            config["hidden_size"],

        num_layers=
            config["num_layers"],

        pred_len=
            config["pred_len"],

        output_size=
            train_dataset.num_features,

        dropout=
            config["dropout"],

    )


    model.to(device)



    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )



    criterion = nn.MSELoss()



    optimizer = AdamW(

        model.parameters(),

        lr=
            config["learning_rate"],

        weight_decay=
            config["weight_decay"],

    )



    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

    )



    early_stopping = EarlyStopping(

        patience=
            config["early_stopping_patience"],

        min_delta=1e-5,

    )



    result_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    best_model_path = (

        result_dir

        /

        "best_model.pt"

    )



    history = []



    # ========================================================
    # Training
    # ========================================================


    for epoch in range(

        1,

        config["max_epochs"] + 1,

    ):



        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------


        model.train()


        train_loss_sum = 0.0

        train_samples = 0



        for x, y in train_loader:


            x = x.to(device)

            y = y.to(device)



            optimizer.zero_grad(

                set_to_none=True

            )



            prediction = model(x)



            loss = criterion(

                prediction,

                y,

            )



            loss.backward()



            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                config["grad_clip"],

            )



            optimizer.step()



            batch_size = x.size(0)



            train_loss_sum += (

                loss.item()

                *

                batch_size

            )


            train_samples += batch_size



        train_loss = (

            train_loss_sum

            /

            train_samples

        )



        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------


        model.eval()


        val_loss_sum = 0.0

        val_samples = 0



        with torch.inference_mode():


            for x, y in val_loader:


                x = x.to(device)

                y = y.to(device)



                prediction = model(x)



                loss = criterion(

                    prediction,

                    y,

                )



                batch_size = x.size(0)



                val_loss_sum += (

                    loss.item()

                    *

                    batch_size

                )


                val_samples += batch_size



        val_loss = (

            val_loss_sum

            /

            val_samples

        )



        scheduler.step(

            val_loss

        )



        current_lr = (

            optimizer

            .param_groups[0]["lr"]

        )



        improved, should_stop = (

            early_stopping.step(

                val_loss

            )

        )



        # ====================================================
        # Save best checkpoint
        # ====================================================


        if improved:


            checkpoint = {


                "epoch":

                    epoch,


                "model_name":

                    "LSTM",


                "model_state_dict":

                    model.state_dict(),


                "val_loss":

                    val_loss,


                "input_size":

                    train_dataset.num_features,


                "config":

                    config,

            }



            # 如果Dataset存在feature_columns则保存

            if hasattr(
                train_dataset,
                "feature_columns"
            ):

                checkpoint[
                    "feature_columns"
                ] = train_dataset.feature_columns



            torch.save(

                checkpoint,

                best_model_path,

            )



        history.append({

            "epoch":

                epoch,


            "train_loss":

                train_loss,


            "val_loss":

                val_loss,


            "learning_rate":

                current_lr,


            "best":

                improved,

        })



        print(

            f"Epoch [{epoch:02d}/"

            f"{config['max_epochs']}] "

            f"| Train: {train_loss:.6f} "

            f"| Val: {val_loss:.6f} "

            f"| LR: {current_lr:.2e}"

        )



        if should_stop:

            break



    # ========================================================
    # Save history
    # ========================================================


    history_df = pd.DataFrame(

        history

    )


    history_df.to_csv(

        result_dir

        /

        "training_history.csv",

        index=False,

    )



    return {


        "model":

            "LSTM",


        "best_val_loss":

            early_stopping.best_loss,


        "best_epoch":

            int(

                history_df.loc[

                    history_df[

                        "val_loss"

                    ].idxmin(),

                    "epoch"

                ]

            ),


        "epochs_run":

            len(history_df),


        "trainable_params":

            trainable_params,

    }