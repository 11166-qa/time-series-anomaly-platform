import torch

from datasets.time_series_dataset import (
    TimeSeriesDataset,
    create_dataloader,
)

from models.lstm import LSTMForecaster


dataset = TimeSeriesDataset(
    csv_path="data/processed/Electricity/test.csv",
    context_csv_path="data/processed/Electricity/val.csv",
    lookback=336,
    pred_len=24,
)


loader = create_dataloader(
    dataset,
    batch_size=2,
    shuffle=False,
)


x,y = next(iter(loader))


print("x:",x.shape)
print("y:",y.shape)


ck=torch.load(
    "results/Electricity/LSTM/seed/seed_42/best_model.pt",
    map_location="cpu"
)


config=ck["config"]


model=LSTMForecaster(

    input_size=321,

    hidden_size=config["hidden_size"],

    num_layers=config["num_layers"],

    pred_len=config["pred_len"],

    output_size=321,

    dropout=config["dropout"],

)


model.load_state_dict(
    ck["model_state_dict"]
)


model.eval()


with torch.no_grad():

    pred=model(x)


print(
    "prediction:",
    pred.shape
)