import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd


class PricePredictionModel(nn.Module):
    def __init__(self, input_size=3, hidden_dim1=64, hidden_dim2=32):
        super(PricePredictionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x):
        return self.model(x)


# 1. 加载模型权重
model = PricePredictionModel()  # 确保模型结构与训练时相同
model.load_state_dict(torch.load("output/sma_model/sma_model.pth"))
model.eval()  # 设置模型为评估模式

# 2. 加载标准化对象
with open("output/sma_model/sma_scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)
with open("output/sma_model/sma_scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)


# 3. 准备新数据并进行预测
def predict_new_data(new_data):
    """
    使用加载的模型对新数据进行预测
    :param new_data: 包含 'SMA10', 'SMA25', 'volume' 的 numpy 数组，例如：[[SMA10, SMA25, volume]]
    :return: 预测的 close 值
    """
    required_columns = ["SMA10", "SMA25", "volume"]
    if not all(col in new_data.columns for col in required_columns):
        raise ValueError(f"new_data_df 必须包含列：{required_columns}")

    # 数据标准化
    new_data_scaled = scaler_X.transform(new_data[required_columns])

    # 转换为张量并预测
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(new_data_tensor).numpy()

    # 反标准化
    prediction_original = scaler_y.inverse_transform(prediction)

    new_data["predicted_close"] = prediction_original
    return new_data


file_path = r"merged_csv\ETHUSDT_1m_2024-09-17_to_2024-11-04.csv"
df = pd.read_csv(file_path)
df["SMA10"] = df["close"].rolling(window=10).mean()
df["SMA25"] = df["close"].rolling(window=25).mean()
df = df.dropna().reset_index(drop=True)

predicted_close = predict_new_data(df)
predicted_close.to_csv("output/sma_model/predict/eth.csv", index=False)
print("预测的收盘价：", predicted_close)
