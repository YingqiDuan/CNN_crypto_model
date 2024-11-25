import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle


# 1. 数据预处理
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # 计算 SMA
    df["SMA10"] = df["close"].rolling(window=10).mean()
    df["SMA25"] = df["close"].rolling(window=25).mean()
    df = df.dropna().reset_index(drop=True)

    # 提取特征和目标变量
    X = df[["SMA10", "SMA25", "volume"]].values
    y = df["close"].values

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 标准化
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    # 转换为张量
    X_train_tensor, X_test_tensor = torch.tensor(
        X_train, dtype=torch.float32
    ), torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor, y_test_tensor = torch.tensor(
        y_train, dtype=torch.float32
    ), torch.tensor(y_test, dtype=torch.float32)

    return (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        y_test_tensor,
        scaler_X,
        scaler_y,
    )


# 2. 定义模型
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


# 3. 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=100, patience=10):
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算平均训练损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early Stop")
            break


# 4. 评估函数
def evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
        y_test = y_test_tensor.cpu().numpy()

        # 反标准化
        y_pred_original = scaler_y.inverse_transform(y_pred)
        y_test_original = scaler_y.inverse_transform(y_test)

    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label="Actual")
    plt.plot(y_pred_original, label="Predicted")
    plt.legend()
    plt.xlabel("Sample")
    plt.ylabel("Close")
    plt.title("Compare Actual vs Predicted")
    plt.show()


# 5. 主流程
def main(file_path):
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_X, scaler_y = (
        load_and_preprocess_data(file_path)
    )

    # 定义数据加载器
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True
    )

    # 初始化模型、损失函数和优化器
    model = PricePredictionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=100, patience=10)

    # 评估模型
    evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y)

    torch.save(model.state_dict(), "output/sma_model/sma_model.pth")

    with open("output/sma_model/sma_scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
    with open("output/sma_model/sma_scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)


# 运行主程序
file_path = r"merged_csv\BTCUSDT_1m_2023-10-29_to_2024-10-26.csv"
main(file_path)
