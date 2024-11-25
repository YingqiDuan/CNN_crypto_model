import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib  # 用于模型保存与加载


# 1. 数据获取与准备
def load_data(file_path):
    """
    从CSV文件读取股票数据，处理日期和数据类型。
    """
    try:
        # 读取CSV时指定正确的列名，并处理多余的列
        df = pd.read_csv(
            file_path,
            parse_dates=["open_time"],
            infer_datetime_format=True,
            low_memory=False,
            usecols=["open_time", "open", "high", "low", "close", "volume"],
        )
        df.rename(columns={"open_time": "Date"}, inplace=True)
        print(f"数据从 {file_path} 加载完成。")
        print(df.head())
        print(df.info())
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return pd.DataFrame()


def merge_data(original_df, new_df):
    combined_df = (
        pd.concat([original_df, new_df])
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    print("原始数据和新数据已合并。")
    return combined_df


# 2. 特征工程
def feature_engineering(df):
    """
    计算EMA指标并生成交易信号。
    """
    if "close" not in df.columns:
        print("DataFrame中缺少'close'列。")
        return df

    # 确保'close'列为数值类型
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # 计算短期EMA和长期EMA
    ema_short_span = 10
    ema_long_span = 25

    # 如果数据量不足以计算长期EMA，调整EMA周期
    if len(df) < ema_long_span + 5:  # 加上future_days=5
        ema_short_span = int(len(df) * 0.1)  # 例如，取数据长度的10%
        ema_long_span = int(len(df) * 0.3)  # 取数据长度的30%
        print(
            f"数据量不足，调整EMA周期为短期EMA={ema_short_span}，长期EMA={ema_long_span}"
        )

    df["EMA_short"] = df["close"].ewm(span=ema_short_span, adjust=False).mean()
    df["EMA_long"] = df["close"].ewm(span=ema_long_span, adjust=False).mean()

    # 生成交易信号
    df["Signal"] = np.where(
        df["EMA_short"] > df["EMA_long"],
        1,
        np.where(df["EMA_short"] < df["EMA_long"], -1, 0),
    )
    print("特征工程完成。")
    print(df[["Date", "close", "EMA_short", "EMA_long", "Signal"]].head())
    return df


# 3. 标签定义
def define_target(df, future_days=5):
    """
    定义目标变量：未来n天内价格是否上涨。
    """
    if "close" not in df.columns:
        print("DataFrame中缺少'close'列。")
        return df

    df["Future_Return"] = df["close"].shift(-future_days) / df["close"] - 1
    df["Target"] = np.where(df["Future_Return"] > 0, 1, 0)

    # 检查有多少非NaN的Target
    initial_length = len(df)
    df.dropna(subset=["Future_Return"], inplace=True)
    final_length = len(df)
    print(f"目标变量定义完成。删除了 {initial_length - final_length} 行NaN值。")
    print(df[["Date", "close", "Future_Return", "Target"]].tail())
    return df


# 4. 数据分割
def split_data(df, train_size=0.7, val_size=0.15):
    """
    按时间顺序分割数据为训练集、验证集和测试集。
    """
    total_len = len(df)
    if total_len == 0:
        print("DataFrame为空，无法分割数据。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train_end = int(total_len * train_size)
    val_end = train_end + int(total_len * val_size)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"数据分割完成：训练集={len(train)}, 验证集={len(val)}, 测试集={len(test)}")
    return train, val, test


# 5. 模型训练
def train_xgboost(X_train, y_train, X_val, y_val):
    """
    训练XGBoost模型，并在验证集上评估。
    """
    if len(X_train) == 0 or len(X_val) == 0:
        print("训练集或验证集为空，无法训练模型。")
        return None

    # 初始化XGBoost分类器
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 在验证集上预测
    y_pred = model.predict(X_val)

    # 检查y_val和y_pred中的唯一类别
    unique_y_val = np.unique(y_val)
    unique_y_pred = np.unique(y_pred)
    print(f"Validation Set Classes: {unique_y_val}")
    print(f"Predicted Classes: {unique_y_pred}")

    # 定义所有可能的标签
    labels = [0, 1]
    target_names = ["Down", "Up"]

    # 评估模型
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred, labels=labels))
    print("\nValidation Classification Report:")
    print(
        classification_report(
            y_val, y_pred, labels=labels, target_names=target_names, zero_division=0
        )
    )

    # ROC AUC
    try:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"Validation AUC: {auc:.4f}")

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Train ROC Curve")
        plt.legend()
        plt.savefig("train_roc_curve.png")
        plt.show()

    except ValueError as ve:
        print(f"ROC AUC计算失败: {ve}")

    return model


# 6. 超参数调优
def hyperparameter_tuning(X_train, y_train):
    """
    使用GridSearchCV对XGBoost进行超参数调优。
    """
    if len(X_train) == 0:
        print("训练集为空，无法进行超参数调优。")
        return None

    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
        param_grid=param_grid,
        cv=3,  # 使用3折交叉验证，加快搜索速度
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print("最佳参数:", grid_search.best_params_)
    print("最佳F1分数:", grid_search.best_score_)

    return grid_search.best_estimator_


# 7. 模型评估
def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    评估模型性能。
    """
    if model is None:
        print("模型为空，无法评估。")
        return

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    unique_y = np.unique(y)
    unique_y_pred = np.unique(y_pred)
    print(f"\n{dataset_name} Classes: {unique_y}")
    print(f"{dataset_name} Predicted Classes: {unique_y_pred}")

    labels = [0, 1]
    target_names = ["Down", "Up"]

    print(f"\n{dataset_name} Confusion Matrix:")
    print(confusion_matrix(y, y_pred, labels=labels))
    print(f"\n{dataset_name} Classification Report:")
    print(
        classification_report(
            y, y_pred, labels=labels, target_names=target_names, zero_division=0
        )
    )

    try:
        auc = roc_auc_score(y, y_proba)
        print(f"{dataset_name} AUC: {auc:.4f}")

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset_name} ROC Curve")
        plt.legend()
        plt.savefig(f"{dataset_name}_roc_curve.png")
        plt.show()
    except ValueError as ve:
        print(f"ROC AUC计算失败: {ve}")


# 8. 模型保存
def save_model(model, filename):
    """
    使用joblib保存模型。
    """
    if model is None:
        print("模型为空，无法保存。")
        return
    joblib.dump(model, filename)
    print(f"模型已保存到 {filename}")


# 9. 模型加载
def load_model(filename):
    """
    加载已保存的模型。
    """
    try:
        model = joblib.load(filename)
        print(f"模型已从 {filename} 加载。")
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None


# 10. 策略回测
def backtest_strategy(model, test, features):
    """
    基于模型预测信号进行策略回测。
    """
    if model is None:
        print("模型为空，无法进行回测。")
        return

    X_test = test[features]
    y_test = test["Target"]

    # 预测
    y_test_pred = model.predict(X_test)

    # 将预测信号添加到测试集中
    test = test.copy()
    test["Predicted_Signal"] = y_test_pred

    # 计算每日市场收益率
    test["Market_Return"] = test["close"].pct_change()

    # 计算策略收益率：使用前一天的信号进行交易
    test["Strategy_Return"] = test["Predicted_Signal"].shift(1) * test["Market_Return"]

    # 计算累计收益率
    test["Cumulative_Market_Return"] = (1 + test["Market_Return"]).cumprod()
    test["Cumulative_Strategy_Return"] = (1 + test["Strategy_Return"]).cumprod()

    # 绘制累计收益率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(test["Cumulative_Market_Return"], label="Market yield")
    plt.plot(test["Cumulative_Strategy_Return"], label="Strategy yield")
    plt.xlabel("Days")
    plt.ylabel("Cumulative yield")
    plt.title("Strategy vs Market yield")
    plt.legend()
    plt.savefig("Strategy vs Market yield.png")
    plt.show()

    # 计算总收益率
    total_market_return = test["Cumulative_Market_Return"].iloc[-1]
    total_strategy_return = test["Cumulative_Strategy_Return"].iloc[-1]
    print(f"总市场收益率: {total_market_return:.2f}")
    print(f"总策略收益率: {total_strategy_return:.2f}")

    # 年化收益率
    def calculate_annual_return(cum_return, periods):
        return cum_return ** (252 / periods) - 1

    market_periods = len(test)
    strategy_periods = len(test)

    annual_market_return = calculate_annual_return(total_market_return, market_periods)
    annual_strategy_return = calculate_annual_return(
        total_strategy_return, strategy_periods
    )

    print(f"年化市场收益率: {annual_market_return:.2%}")
    print(f"年化策略收益率: {annual_strategy_return:.2%}")

    # 夏普比率
    def calculate_sharpe_ratio(strategy_returns, risk_free_rate=0):
        excess_returns = strategy_returns - risk_free_rate / 252
        return (
            np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            if excess_returns.std() != 0
            else 0
        )

    sharpe_market = calculate_sharpe_ratio(test["Market_Return"])
    sharpe_strategy = calculate_sharpe_ratio(test["Strategy_Return"])

    print(f"市场夏普比率: {sharpe_market:.2f}")
    print(f"策略夏普比率: {sharpe_strategy:.2f}")
