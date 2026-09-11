from pathlib import Path
import time







from trainers.transformer_trainer import (
    train_transformer,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ETTh1"
    / "Transformer"
    / "baseline"
)


CONFIG = {

    # ========================================================
    # 时序任务
    # ========================================================

    "lookback": 336,
    "pred_len": 24,

    # ========================================================
    # Transformer
    # ========================================================

    "d_model": 64,
    "nhead": 4,

    "num_layers": 2,

    "dim_feedforward": 128,

    "dropout": 0.1,

    # ========================================================
    # Training
    # ========================================================

    # Transformer CPU训练先采用32
    "batch_size": 32,

    "learning_rate": 1e-3,

    "weight_decay": 1e-4,

    "max_epochs": 30,

    "early_stopping_patience": 8,

    "grad_clip": 1.0,

    "seed": 42,
}


def main():

    print("=" * 70)
    print("ETTh1 Transformer Baseline")
    print("=" * 70)

    start_time = (
        time.perf_counter()
    )

    result = train_transformer(
        data_dir=DATA_DIR,
        result_dir=RESULT_DIR,
        config=CONFIG,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Transformer Baseline Finished"
    )

    print(
        "=" * 70
    )

    print(
        f"Best Val Loss: "
        f"{result['best_val_loss']:.6f}"
    )

    print(
        f"Best Epoch: "
        f"{result['best_epoch']}"
    )

    print(
        f"Epochs Run: "
        f"{result['epochs_run']}"
    )

    print(
        f"Parameters: "
        f"{result['trainable_params']:,}"
    )

    print(
        f"Training Time: "
        f"{elapsed_seconds:.2f} s"
    )


if __name__ == "__main__":
    main()