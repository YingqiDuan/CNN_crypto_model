import os
import pickle
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from binance.um_futures import UMFutures
from get_trading_pairs import coins
from datetime import datetime
from zoneinfo import ZoneInfo


# the CNN model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 10 * 3, 256)
        self.fc2 = nn.Linear(256, 3)  # Assuming 3 classes

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))  # Pool1

        x = F.relu(self.bn2(self.conv2(x)))  # Conv2
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))  # Pool2

        x = F.relu(self.bn3(self.conv3(x)))  # Conv3
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))  # Pool3

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # FC1
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # FC2
        return x


def get_data(client, coin, interval, candlesticks):
    """
    获取指定交易对的连续K线数据。

    :param client: UMFutures 客户端实例
    :param coin: 交易对符号，例如 'BTCUSDT'
    :param interval: K线间隔，例如 '4h'
    :param candlesticks: 需要获取的天数
    :return: DataFrame 格式的K线数据
    """
    try:
        klines = client.continuous_klines(
            coin, "PERPETUAL", interval, limit=candlesticks
        )
        time.sleep(0.1)
        df = pd.DataFrame(klines)
        return df
    except Exception as e:
        print(f"Error fetching data for {coin}: {e}")
        return pd.DataFrame()


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

    # 正则化
    df = df / df.iloc[0]

    return df


def load_model(model_path, device):
    """
    Load the trained CNN model.

    Parameters:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model on.

    Returns:
        nn.Module: Loaded CNN model.
    """
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def load_scaler(scaler_path):
    """
    Load the saved StandardScaler.

    Parameters:
        scaler_path (str): Path to the saved scaler file.

    Returns:
        StandardScaler: Loaded scaler.
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")
    return scaler


def preprocess_sample(sample, scaler, candlesticks):
    """
    Preprocess the input sample by standardizing it.

    Parameters:
        sample (numpy.ndarray): Input sample with shape (candlesticks, 9).
        scaler (StandardScaler): Fitted scaler.

    Returns:
        torch.Tensor: Preprocessed sample tensor ready for prediction.
    """
    if sample.shape != (candlesticks, 9):
        raise ValueError(
            f"Expected sample shape ({candlesticks}, 9), got {sample.shape}"
        )

    # Flatten and standardize
    sample_flat = sample.reshape(1, -1)  # Shape: (1, 900)
    sample_scaled = scaler.transform(sample_flat)  # Shape: (1, 900)
    sample_scaled = sample_scaled.reshape(
        1, candlesticks, 9
    )  # Shape: (1, candlesticks, 9)

    # Convert to tensor and add channel dimension
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).unsqueeze(
        0
    )  # Shape: (1, 1, candlesticks, 9)
    return sample_tensor


def predict(model, sample_tensor, device):
    """
    Predict the class of a single sample.

    Parameters:
        model (nn.Module): Loaded CNN model.
        sample_tensor (torch.Tensor): Preprocessed sample tensor.
        device (torch.device): Device for computation.

    Returns:
        int: Predicted class label.
        numpy.ndarray: Probability scores for each class.
    """
    sample_tensor = sample_tensor.to(device)
    with torch.no_grad():
        outputs = model(sample_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
    return predicted_label, probabilities


def main():
    # Paths (update these paths as necessary)
    model_path = "cnn_model_4h.pth"  # Path to your trained model
    scaler_path = "scaler.pkl"  # Path to your saved scaler

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    if not os.path.exists(scaler_path):
        print(f"Scaler file not found at {scaler_path}")
        return

    interval = "4h"
    candlesticks = 100  # 需要获取的个数

    # 初始化 UMFutures 客户端
    try:
        um_futures_client = UMFutures()
    except Exception as e:
        print(f"Error initializing UMFutures client: {e}")
        return

    # 定义CSV文件路径
    csv_file = "predictions.csv"

    # 获取当前时间作为预测时间
    pacific_tz = ZoneInfo("US/Pacific")
    prediction_time = datetime.now(pacific_tz).isoformat()

    # 检查CSV文件是否存在
    if os.path.exists(csv_file):
        df_predictions = pd.read_csv(csv_file, index_col=0)
    else:
        # 初始化DataFrame，索引为交易对
        df_predictions = pd.DataFrame(index=coins)
    df_predictions.index.name = None

    # 添加当前预测时间作为新的列
    df_predictions[prediction_time] = 0  # 先填充0，后面再赋值

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and scaler
    model = load_model(model_path, device)
    scaler = load_scaler(scaler_path)

    for coin in coins:
        # 获取数据
        df = get_data(um_futures_client, coin, interval, candlesticks)
        if df.empty:
            print(f"No data fetched for {coin}. Skipping...")
            continue

        # 检查数据行数是否至少为days
        if len(df) < candlesticks:
            print(f"Not enough data for {coin} (found {len(df)} rows). Skipping...")
            continue

        # 清洗数据
        df = clean_data(df)
        sample = df.to_numpy()

        # Preprocess the sample
        try:
            sample_tensor = preprocess_sample(sample, scaler, candlesticks)
        except ValueError as e:
            print(e)
            return

        # Make prediction
        predicted_label, probabilities = predict(model, sample_tensor, device)

        # Map the predicted label back to original classes
        # Assuming during training labels were mapped as {-1: 0, 0: 1, 1: 2}
        label_mapping = {0: -1, 1: 0, 2: 1}
        original_label = label_mapping.get(predicted_label, "Unknown")
        df_predictions.at[coin, prediction_time] = original_label

        if np.any(probabilities > 0.8):
            print(f"{coin} predicted: {original_label}, probability: {probabilities}")

    df_predictions.to_csv(csv_file, index=True, index_label=None)


if __name__ == "__main__":
    main()
