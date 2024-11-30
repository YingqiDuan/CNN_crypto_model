from cnn import (
    standardize_samples,
    load_model,
    predict,
)
import torch
from binance.um_futures import UMFutures
import pandas as pd
import numpy as np
import time
from get_trading_pairs import coins


def get_data(client, coin, interval, days):
    """
    获取指定交易对的连续K线数据。

    :param client: UMFutures 客户端实例
    :param coin: 交易对符号，例如 'BTCUSDT'
    :param interval: K线间隔，例如 '4h'
    :param days: 需要获取的天数
    :return: DataFrame 格式的K线数据
    """
    try:
        klines = client.continuous_klines(coin, "PERPETUAL", interval, limit=days)
        time.sleep(0.1)  # 防止API速率限制
        df = pd.DataFrame(klines)
        return df
    except Exception as e:
        print(f"Error fetching data for {coin}: {e}")
        return pd.DataFrame()  # 返回空DataFrame以避免后续错误


def clean_data(df):
    """
    清洗和处理K线数据。

    :param df: 原始K线DataFrame
    :return: 清洗后的DataFrame
    """
    if df.empty:
        return df

    df.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qav",
        "num_trades",
        "taker_base_vol",
        "taker_quote_vol",
        "ignore",
    ]

    # 转换时间戳为日期时间
    for time_col in ["open_time", "close_time"]:
        df[time_col] = pd.to_datetime(df[time_col], unit="ms")
        df[time_col] = df[time_col].dt.tz_localize(
            "UTC", ambiguous="NaT", nonexistent="NaT"
        )
        df[time_col] = df[time_col].dt.tz_convert("US/Pacific")

    # 设置索引
    df.set_index("open_time", inplace=True)
    df = df.tz_localize(None)

    # 转换数据类型
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "qav",
        "num_trades",
        "taker_base_vol",
        "taker_quote_vol",
    ]
    df[numeric_columns] = df[numeric_columns].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)

    # 删除不需要的列
    df.drop(
        columns=["open_time", "close_time", "ignore"], inplace=True, errors="ignore"
    )

    return df


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model_path = "cnn_model_4h.pth"
    try:
        model = load_model(model_path, device)
        model.eval()  # 设置模型为评估模式
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    interval = "4h"
    days = 100  # 需要获取的个数

    # 初始化 UMFutures 客户端
    try:
        um_futures_client = UMFutures()
    except Exception as e:
        print(f"Error initializing UMFutures client: {e}")
        return

    for coin in coins:
        # 获取数据
        df = get_data(um_futures_client, coin, interval, days)
        if df.empty:
            print(f"No data fetched for {coin}. Skipping...")
            continue

        # 清洗数据
        df = clean_data(df)
        if df.empty:
            print(f"No valid data after cleaning for {coin}. Skipping...")
            continue

        # 检查数据行数是否至少为days
        if len(df) < days:
            print(f"Not enough data for {coin} (found {len(df)} rows). Skipping...")
            continue

        df = df.to_numpy()
        df = df.reshape(1, days, 9)

        # 标准化样本
        try:
            sample = standardize_samples(df)
        except Exception as e:
            print(f"Error standardizing samples for {coin}: {e}")
            continue

        predicted_label = predict(model, sample, device)

        if predicted_label == 1 or predicted_label == -1:
            print(f"{coin} 预测: {predicted_label}")


if __name__ == "__main__":
    main()
