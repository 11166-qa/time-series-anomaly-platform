from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ETTh1.csv"


def inspect_data():
    """检查 ETTh1 数据集的基本结构和数据质量。"""

    # ========================================================
    # 2. 检查文件是否存在
    # ========================================================

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}"
        )

    print("=" * 70)
    print("ETTh1 Data Inspection")
    print("=" * 70)

    # ========================================================
    # 3. 读取数据
    # ========================================================

    df = pd.read_csv(DATA_PATH)

    print("\n[1] 数据基本信息")
    print("-" * 70)

    print(f"数据路径：{DATA_PATH}")
    print(f"数据形状：{df.shape}")
    print(f"数据行数：{df.shape[0]}")
    print(f"数据列数：{df.shape[1]}")
    print(f"列名：{df.columns.tolist()}")

    print("\n前 5 行：")
    print(df.head())

    print("\n各列数据类型：")
    print(df.dtypes)

    # ========================================================
    # 4. 缺失值检查
    # ========================================================

    print("\n[2] 缺失值检查")
    print("-" * 70)

    missing_count = df.isna().sum()

    missing_rate = (
        df.isna().mean() * 100
    ).round(4)

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_rate_%": missing_rate
    })

    print(missing_report)

    print(
        f"\n总缺失值数量："
        f"{df.isna().sum().sum()}"
    )

    # ========================================================
    # 5. 重复值检查
    # ========================================================

    print("\n[3] 重复数据检查")
    print("-" * 70)

    duplicate_rows = df.duplicated().sum()

    print(f"完全重复行数量：{duplicate_rows}")

    # ========================================================
    # 6. 时间字段处理
    # ========================================================

    print("\n[4] 时间字段检查")
    print("-" * 70)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    invalid_date_count = df["date"].isna().sum()

    print(f"无法解析的时间数量：{invalid_date_count}")

    print(f"开始时间：{df['date'].min()}")
    print(f"结束时间：{df['date'].max()}")

    print(
        f"时间是否严格按升序排列："
        f"{df['date'].is_monotonic_increasing}"
    )

    duplicate_timestamp = (
        df["date"].duplicated().sum()
    )

    print(
        f"重复时间戳数量："
        f"{duplicate_timestamp}"
    )

    # ========================================================
    # 7. 时间间隔检查
    # ========================================================

    time_diff = df["date"].diff().dropna()

    print("\n主要采样间隔：")
    print(time_diff.value_counts().head(10))

    if not time_diff.empty:
        expected_interval = time_diff.mode().iloc[0]

        irregular_count = (
            time_diff != expected_interval
        ).sum()

        print(
            f"\n主要采样间隔："
            f"{expected_interval}"
        )

        print(
            f"非标准采样间隔数量："
            f"{irregular_count}"
        )

    # ========================================================
    # 8. 数值列检查
    # ========================================================

    print("\n[5] 数值变量检查")
    print("-" * 70)

    numeric_columns = (
        df.select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    print(f"数值变量：{numeric_columns}")
    print(f"数值变量数量：{len(numeric_columns)}")

    # ========================================================
    # 9. 无穷值检查
    # ========================================================

    infinity_count = np.isinf(
        df[numeric_columns].to_numpy()
    ).sum()

    print(f"无穷值数量：{infinity_count}")

    # ========================================================
    # 10. 描述性统计
    # ========================================================

    print("\n[6] 描述性统计")
    print("-" * 70)

    print(
        df[numeric_columns]
        .describe()
        .T
        .round(4)
    )

    # ========================================================
    # 11. 最终汇总
    # ========================================================

    print("\n" + "=" * 70)
    print("数据质量检查完成")
    print("=" * 70)

    print(f"样本数量：{len(df)}")
    print(f"数值变量数量：{len(numeric_columns)}")
    print(f"缺失值数量：{df.isna().sum().sum()}")
    print(f"重复行数量：{duplicate_rows}")
    print(f"重复时间戳数量：{duplicate_timestamp}")
    print(f"无穷值数量：{infinity_count}")


if __name__ == "__main__":
    inspect_data()