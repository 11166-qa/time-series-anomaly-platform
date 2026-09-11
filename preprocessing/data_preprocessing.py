from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ETTh1.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ETTh1"
)


# ============================================================
# 2. 数据集划分比例
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


def preprocess_etth1():
    """
    ETTh1 数据预处理：

    1. 读取原始数据
    2. 时间字段转换与排序
    3. 按时间顺序划分 Train / Validation / Test
    4. 仅使用训练集拟合 StandardScaler
    5. 标准化三个数据集
    6. 保存处理结果与 Scaler
    """

    print("=" * 70)
    print("ETTh1 Data Preprocessing")
    print("=" * 70)

    # ========================================================
    # 3. 检查数据文件
    # ========================================================

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到原始数据文件：{RAW_DATA_PATH}"
        )

    # ========================================================
    # 4. 读取原始数据
    # ========================================================

    df = pd.read_csv(RAW_DATA_PATH)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    # 即使数据已经有序，也显式排序
    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(f"\n原始数据形状：{df.shape}")

    # ========================================================
    # 5. 数据质量保护
    # ========================================================

    if df.isna().any().any():
        raise ValueError(
            "数据中存在缺失值，请先完成缺失值处理。"
        )

    numeric_columns = [
        col
        for col in df.columns
        if col != "date"
    ]

    numeric_values = (
        df[numeric_columns]
        .to_numpy()
    )

    if np.isinf(numeric_values).any():
        raise ValueError(
            "数值变量中存在无穷值。"
        )

    print(f"数值变量：{numeric_columns}")
    print(f"数值变量数量：{len(numeric_columns)}")

    # ========================================================
    # 6. 按时间顺序划分数据
    # ========================================================

    total_size = len(df)

    train_size = int(
        total_size * TRAIN_RATIO
    )

    val_size = int(
        total_size * VAL_RATIO
    )

    test_size = (
            total_size
            - train_size
            - val_size
    )

    train_end = train_size
    val_end = train_size + val_size
    print(
        f"划分数量检查："
        f"Train={train_size}, "
        f"Val={val_size}, "
        f"Test={test_size}"
    )

    train_df = (
        df.iloc[:train_end]
        .copy()
    )

    val_df = (
        df.iloc[train_end:val_end]
        .copy()
    )

    test_df = (
        df.iloc[val_end:]
        .copy()
    )

    print("\n" + "-" * 70)
    print("数据集划分")
    print("-" * 70)

    print(
        f"Train：{len(train_df)} 条，"
        f"{train_df['date'].min()} "
        f"→ {train_df['date'].max()}"
    )

    print(
        f"Validation：{len(val_df)} 条，"
        f"{val_df['date'].min()} "
        f"→ {val_df['date'].max()}"
    )

    print(
        f"Test：{len(test_df)} 条，"
        f"{test_df['date'].min()} "
        f"→ {test_df['date'].max()}"
    )

    # ========================================================
    # 7. 创建 StandardScaler
    # ========================================================

    scaler = StandardScaler()

    # 注意：
    # 这里只允许 fit 训练集
    scaler.fit(
        train_df[numeric_columns]
    )

    # ========================================================
    # 8. 分别进行标准化
    # ========================================================

    train_scaled = scaler.transform(
        train_df[numeric_columns]
    )

    val_scaled = scaler.transform(
        val_df[numeric_columns]
    )

    test_scaled = scaler.transform(
        test_df[numeric_columns]
    )

    # 使用 float32
    train_scaled = train_scaled.astype(
        np.float32
    )

    val_scaled = val_scaled.astype(
        np.float32
    )

    test_scaled = test_scaled.astype(
        np.float32
    )

    # ========================================================
    # 9. 写回 DataFrame
    # ========================================================

    train_df.loc[
        :, numeric_columns
    ] = train_scaled

    val_df.loc[
        :, numeric_columns
    ] = val_scaled

    test_df.loc[
        :, numeric_columns
    ] = test_scaled

    # ========================================================
    # 10. 创建输出目录
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 11. 保存处理后的数据
    # ========================================================

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "val.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # ========================================================
    # 12. 保存 StandardScaler
    # ========================================================

    scaler_path = (
        OUTPUT_DIR
        / "scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    # ========================================================
    # 13. 保存元信息
    # ========================================================

    metadata = {
        "dataset": "ETTh1",
        "total_samples": total_size,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),

        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,

        "numeric_columns": numeric_columns,

        "train_start": str(
            train_df["date"].min()
        ),
        "train_end": str(
            train_df["date"].max()
        ),

        "val_start": str(
            val_df["date"].min()
        ),
        "val_end": str(
            val_df["date"].max()
        ),

        "test_start": str(
            test_df["date"].min()
        ),
        "test_end": str(
            test_df["date"].max()
        )
    }

    metadata_path = (
        OUTPUT_DIR
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # 14. 输出标准化检查
    # ========================================================

    print("\n" + "-" * 70)
    print("标准化结果检查")
    print("-" * 70)

    print("\nTrain 标准化后均值：")

    print(
        train_df[numeric_columns]
        .mean()
        .round(4)
    )

    print("\nTrain 标准化后标准差：")

    print(
        train_df[numeric_columns]
        .std()
        .round(4)
    )

    print("\n" + "=" * 70)
    print("预处理完成")
    print("=" * 70)

    print(f"Train：{train_path}")
    print(f"Validation：{val_path}")
    print(f"Test：{test_path}")
    print(f"Scaler：{scaler_path}")
    print(f"Metadata：{metadata_path}")


if __name__ == "__main__":
    preprocess_etth1()