import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import optuna
import joblib
import matplotlib.pyplot as plt


# 1. 数据加载和预处理函数
def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(
            file_path,
            parse_dates=["open_time"],  # 使用正确的时间列名
            dtype={
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
                "quote_volume": "float64",
                "count": "float64",
                "taker_buy_volume": "float64",
                "taker_buy_quote_volume": "float64",
                # 不需要的列将被删除，因此不指定它们的dtype
            },
            low_memory=False,  # 避免低内存时的类型推断
        )
    except ValueError as e:
        print(f"读取 CSV 文件时出错: {e}")
        return None
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None

    # 确保按时间排序
    if "open_time" in data.columns:
        data = data.sort_values("open_time").reset_index(drop=True)
    else:
        print("警告: 数据中缺少 'open_time' 列。请检查列名。")

    # 删除不需要的列
    columns_to_drop = ["close_time", "ignore", "source_file"]
    data = data.drop(columns=columns_to_drop, errors="ignore")
    print(f"已删除列: {columns_to_drop}")

    # 检查剩余列
    print("剩余列名:", data.columns.tolist())

    # 计算技术指标
    data = add_technical_indicators(data)

    # 处理缺失值
    data = handle_missing_values(data)

    # 清洗整数列（如果需要，可以跳过这一步，因为已经删除了部分列）
    # data = clean_integer_columns(data)

    # 计算交易信号
    data["Signal"] = np.where(data["EMA_12"] > data["EMA_26"], 1, -1)
    data["Cross"] = data["Signal"].diff()
    data["Label"] = np.where(data["Cross"] > 0, 1, np.where(data["Cross"] < 0, -1, 0))

    # 检查标签分布
    label_counts = data["Label"].value_counts()
    print("标签分布:\n", label_counts)

    # 处理可能存在的 NaN
    data = data.dropna()
    print("数据形状 after dropna:", data.shape)
    return data


def add_technical_indicators(data):
    # EMA
    data["EMA_12"] = data["close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["close"].ewm(span=26, adjust=False).mean()

    # MACD
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = data["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    RS = roll_up / roll_down
    data["RSI"] = 100.0 - (100.0 / (1.0 + RS))

    # Bollinger Bands
    data["SMA_20"] = data["close"].rolling(window=20).mean()
    data["STD_20"] = data["close"].rolling(window=20).std()
    data["Bollinger_Upper"] = data["SMA_20"] + (data["STD_20"] * 2)
    data["Bollinger_Lower"] = data["SMA_20"] - (data["STD_20"] * 2)

    return data


def handle_missing_values(data):
    # 检查缺失值
    missing = data.isnull().sum()
    print("缺失值统计:\n", missing)

    # 选择填充或删除缺失值
    # 例如，填充前向值
    data = data.fillna(method="ffill")

    # 也可以选择删除仍然存在缺失值的行
    data = data.dropna()

    return data


# 2. 构造特征和标签
def construct_features_and_labels(data):
    # 添加更多技术指标作为特征
    features = [
        "EMA_12",
        "EMA_26",
        "close",
        "volume",
        "MACD",
        "Signal_Line",
        "RSI",
        "SMA_20",
        "Bollinger_Upper",
        "Bollinger_Lower",
    ]
    X = data[features].shift(1).dropna()
    y = data["Label"].shift(-1).dropna()

    # 对齐索引
    X = X.loc[y.index]

    print("特征形状:", X.shape)
    print("标签形状:", y.shape)

    return X, y


# 3. 定义模型训练和评估的函数
def objective(trial, X_train, X_valid, y_train, y_valid):
    # 定义需要优化的超参数
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": 42,
    }
    # 构建管道
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", RandomForestClassifier(**params))]
    )
    # 训练模型
    pipeline.fit(X_train, y_train)
    # 预测验证集
    y_pred = pipeline.predict(X_valid)
    # 计算准确率
    accuracy = accuracy_score(y_valid, y_pred)
    return accuracy


# 4. 主函数
def main():
    # 项目路径和数据路径
    project_path = r"D:\crypto_ml_project\My Project"
    data_path = os.path.join(
        project_path, "merged_csv", "DOGEUSDT_4h_2023-11-23_to_2024-11-20.csv"
    )
    model_dir = os.path.join(project_path, "models")
    os.makedirs(model_dir, exist_ok=True)  # 确保模型目录存在

    # 加载数据
    data = load_and_preprocess_data(data_path)
    if data is None or data.empty:
        print("Error: 数据加载失败或数据为空。")
        return

    # 构造特征和标签
    X, y = construct_features_and_labels(data)

    # 检查 X 和 y 是否为空
    if X.empty or y.empty:
        print("Error: 特征或标签数据为空。请检查数据处理步骤。")
        return

    # 检查标签是否仅包含 0
    unique_labels = y.unique()
    if set(unique_labels) <= {0}:
        print("Error: 标签数据仅包含 0，没有有效的交易信号（1 或 -1）。")
        return

    # 切分数据集
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("训练集和验证集形状:", X_train_val.shape, y_train_val.shape)
        print("测试集形状:", X_test.shape, y_test.shape)
    except ValueError as e:
        print(f"切分数据集时出错: {e}")
        return

    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=42,
            stratify=y_train_val,
        )  # 0.25 x 0.8 = 0.2
        print("训练集形状:", X_train.shape, y_train.shape)
        print("验证集形状:", X_valid.shape, y_valid.shape)
    except ValueError as e:
        print(f"切分训练集和验证集时出错: {e}")
        return

    # 检查标签分布是否在训练集和验证集中平衡
    print("训练集标签分布:\n", y_train.value_counts())
    print("验证集标签分布:\n", y_valid.value_counts())

    # 使用 Optuna 进行贝叶斯优化
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, X_valid, y_train, y_valid), n_trials=50
    )

    print("最佳超参数: ", study.best_params)
    print("最佳准确率: ", study.best_value)

    # 使用最佳超参数训练最终模型
    best_params = study.best_params
    best_params["random_state"] = 42
    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**best_params)),
        ]
    )
    final_pipeline.fit(X_train_val, y_train_val)

    # 保存模型
    model_path = os.path.join(model_dir, "best_random_forest_model.joblib")
    joblib.dump(final_pipeline, model_path)
    print(f"模型已保存到: {model_path}")

    # 加载并评估模型
    loaded_model = joblib.load(model_path)
    y_pred = loaded_model.predict(X_test)

    # 模型评估
    print(classification_report(y_test, y_pred))
    print("测试集准确率:", accuracy_score(y_test, y_pred))

    # 策略回测
    # 注意：此处为了简化，只使用测试集的索引
    test_indices = X_test.index
    data_test = data.loc[test_indices].copy()  # 防止 SettingWithCopyWarning
    data_test["Predicted_Label"] = y_pred
    data_test["Strategy_Return"] = (
        data_test["Predicted_Label"].shift(1) * data_test["close"].pct_change()
    )
    data_test["Market_Return"] = data_test["close"].pct_change()
    data_test = data_test.dropna()
    cumulative_strategy_return = (1 + data_test["Strategy_Return"]).cumprod()
    cumulative_market_return = (1 + data_test["Market_Return"]).cumprod()

    print("策略累计收益:", cumulative_strategy_return.iloc[-1])
    print("市场累计收益:", cumulative_market_return.iloc[-1])

    # 绘制收益曲线
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_strategy_return, label="Strategy")
    plt.plot(cumulative_market_return, label="Market")
    plt.legend()
    plt.title("策略与市场累计收益对比")
    plt.xlabel("时间")
    plt.ylabel("累计收益")
    plt.show()

    # 保存回测图表
    plot_path = os.path.join(project_path, "strategy_vs_market_return.png")
    plt.savefig(plot_path)
    print(f"收益曲线已保存到: {plot_path}")


if __name__ == "__main__":
    main()
