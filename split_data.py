import pandas as pd
import pickle


def split_data(csv_path, save_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 初始化存储样本和标签的列表
    samples = []
    labels = []

    # 提取不同条件的样本
    for idx in range(100, len(df)):
        # 提取当前索引的前 100 行
        matrix = df.iloc[idx - 100 : idx].reset_index(drop=True)

        # 根据条件设置 label
        if df.loc[idx, "close"] > df.loc[idx, "open"] * 1.01:
            label = 1
        elif df.loc[idx, "close"] < df.loc[idx, "open"] * 0.99:
            label = -1
        else:
            label = 0

        # 保存样本和对应的 label
        samples.append(matrix)
        labels.append(label)

    # 保存样本和标签到文件
    with open(save_path, "wb") as f:
        pickle.dump({"samples": samples, "labels": labels}, f)

    print(f"样本和标签已保存到 {save_path}")


if __name__ == "main":
    csv_path = "merged_csv/BTCUSDT_1h_2023-11-28_to_2024-11-25.csv"
    save_path = "supervised_samples_with_labels.pkl"
    split_data(csv_path, save_path)
