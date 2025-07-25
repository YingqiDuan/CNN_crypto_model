import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn.functional as F
from datetime import datetime
import glob
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler
from cnn import CNN_4h, CNN_1h

# 忽略警告以减少输出噪音
warnings.filterwarnings("ignore")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def clean_data(df):
    """清洗和预处理CSV文件中的数据"""
    # 将时间戳转换为datetime
    df["open_time"] = pd.to_datetime(df["open_time"])

    # 将datetime设为索引
    df = df.set_index("open_time")

    # 确保数值列是浮点数
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def load_model(model_path, model_type="4h", device=device):
    """
    加载训练好的CNN模型

    参数:
        model_path (str): 模型文件路径
        model_type (str): 模型类型，可选 "1h" 或 "4h"
        device (torch.device): 加载模型的设备

    返回:
        nn.Module: 加载的CNN模型
    """
    if model_type == "4h":
        model = CNN_4h()
    elif model_type == "1h":
        model = CNN_1h()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"已从 {model_path} 加载模型")
    return model


def load_scaler(scaler_path):
    """
    加载保存的StandardScaler

    参数:
        scaler_path (str): Scaler文件路径

    返回:
        StandardScaler: 加载的scaler
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"已从 {scaler_path} 加载Scaler")
    return scaler


def prepare_sample(df, lookback=100):
    """
    从DataFrame中准备样本数据

    参数:
        df (DataFrame): 包含OHLCV数据的DataFrame
        lookback (int): 样本长度（默认100个周期）

    返回:
        numpy.ndarray: 形状为(lookback, 9)的样本数据
    """
    # 确保有足够的数据
    if len(df) < lookback:
        raise ValueError(f"数据长度 {len(df)} 小于所需的样本长度 {lookback}")

    # 获取最近的lookback条数据
    recent_data = df.iloc[-lookback:].copy()

    # 获取第一条数据作为基准，进行归一化
    base_values = recent_data.iloc[0]
    normalized_data = recent_data / base_values

    # 构建样本矩阵
    sample = normalized_data[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
    ].values

    return sample


def preprocess_sample(sample, scaler, lookback=100):
    """
    通过标准化来预处理输入样本

    参数:
        sample (numpy.ndarray): 形状为(lookback, 9)的输入样本
        scaler (StandardScaler): 拟合好的scaler
        lookback (int): 样本长度

    返回:
        torch.Tensor: 预处理后的样本张量，准备用于预测
    """
    if sample.shape != (lookback, 9):
        raise ValueError(f"预期样本形状 ({lookback}, 9), 实际得到 {sample.shape}")

    # 展平并标准化
    sample_flat = sample.reshape(1, -1)  # 形状: (1, lookback*9)
    sample_scaled = scaler.transform(sample_flat)  # 形状: (1, lookback*9)
    sample_scaled = sample_scaled.reshape(1, lookback, 9)  # 形状: (1, lookback, 9)

    # 转换为张量并添加通道维度
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).unsqueeze(
        0
    )  # 形状: (1, 1, lookback, 9)
    return sample_tensor


def predict(model, sample_tensor, device=device):
    """
    预测单个样本的类别

    参数:
        model (nn.Module): 加载的CNN模型
        sample_tensor (torch.Tensor): 预处理后的样本张量
        device (torch.device): 计算设备

    返回:
        int: 预测的类别标签 (0=看跌, 1=中性, 2=看涨)
        numpy.ndarray: 每个类别的概率分数
    """
    sample_tensor = sample_tensor.to(device)
    with torch.no_grad():
        outputs = model(sample_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()

    # 将预测标签从(0,1,2)映射回(-1,0,1)
    mapped_label = predicted_label - 1 if predicted_label in [0, 1, 2] else 0

    return mapped_label, probabilities


def generate_signals(
    df, model, scaler, lookback=100, step=1, probability_threshold=0.7
):
    """
    对时间序列数据生成CNN模型预测信号

    参数:
        df (DataFrame): 包含价格数据的DataFrame
        model (nn.Module): 加载的CNN模型
        scaler (StandardScaler): 用于标准化的scaler
        lookback (int): 用于预测的历史数据长度
        step (int): 预测步长(每n条数据进行一次预测)
        probability_threshold (float): 概率阈值，只有当预测概率大于或等于此阈值时才生成信号

    返回:
        DataFrame: 包含预测信号的DataFrame
    """
    # 确保数据长度足够
    if len(df) < lookback:
        return df

    # 创建一个结果DataFrame的副本
    result_df = df.copy()
    result_df["signal"] = 0  # 初始化信号列
    result_df["probability"] = np.nan  # 存储预测概率

    # 从lookback开始，逐步生成预测
    for i in range(lookback, len(df), step):
        try:
            # 提取样本
            sample_window = df.iloc[i - lookback : i]
            sample = prepare_sample(sample_window, lookback)

            # 预处理样本
            sample_tensor = preprocess_sample(sample, scaler, lookback)

            # 预测
            signal, probabilities = predict(model, sample_tensor, device)

            # 获取对应类别的概率 (-1,0,1 -> 0,1,2)
            if signal == -1:
                prob = probabilities[0]
            elif signal == 0:
                prob = probabilities[1]
            else:  # signal == 1
                prob = probabilities[2]

            # 存储预测概率
            result_df.iloc[i, result_df.columns.get_loc("probability")] = prob

            # 只有当概率大于或等于阈值时才生成信号，否则信号为0（不交易）
            if prob >= probability_threshold:
                result_df.iloc[i, result_df.columns.get_loc("signal")] = signal
            else:
                result_df.iloc[i, result_df.columns.get_loc("signal")] = 0

        except Exception as e:
            print(f"在索引{i}处生成预测时出错: {e}")

    return result_df


def backtest(
    df, initial_capital=10000.0, position_size=1.0, fee_rate=0.0005, max_hold_periods=1
):
    """
    回测CNN模型预测的交易策略

    参数:
        df: 带有信号的DataFrame
        initial_capital: 初始资金
        position_size: 每笔交易分配的资金比例(0-1)
        fee_rate: 交易费率
        max_hold_periods: 持仓的最大周期数

    返回:
        DataFrame: 包含回测结果的DataFrame
    """
    # 创建DataFrame副本以避免修改原始数据
    backtest_df = df.copy()

    # 初始化回测列
    backtest_df["position"] = 0  # 0=无仓位, 1=多头, -1=空头
    backtest_df["entry_price"] = 0.0  # 当前仓位的入场价格
    backtest_df["capital"] = initial_capital  # 当前资金
    backtest_df["holdings"] = 0.0  # 当前持仓价值(多头为正，空头为负)
    backtest_df["total_value"] = initial_capital  # 资金+持仓总价值
    backtest_df["hold_periods"] = 0  # 持仓周期数
    backtest_df["forced_close"] = 0  # 因达到最大持仓周期而强制平仓的指标

    position = 0  # 0=无仓位, 1=多头, -1=空头
    entry_price = 0
    capital = initial_capital
    holdings = 0
    hold_periods = 0  # 持仓周期计数器

    # 遍历数据模拟交易
    for i in range(1, len(backtest_df)):
        curr_price = backtest_df["close"].iloc[i]
        signal = backtest_df["signal"].iloc[i]

        # 强制平仓标志
        force_close = False
        if position != 0 and hold_periods >= max_hold_periods:
            force_close = True
            backtest_df.loc[backtest_df.index[i], "forced_close"] = 1

        # 处理信号和强制平仓
        # 情况1: 当前无仓位，遇到买入信号，开多仓
        if signal == 1 and position == 0:
            # 计算仓位大小
            trade_size = capital * position_size
            # 计算手续费
            fee = trade_size * fee_rate
            # 计算扣除手续费后的金额
            amount_after_fees = trade_size - fee
            # 计算买入股数
            shares_bought = amount_after_fees / curr_price
            # 更新仓位
            position = 1
            entry_price = curr_price
            capital -= trade_size
            holdings = shares_bought * curr_price
            hold_periods = 0  # 重置持仓周期计数器

        # 情况2: 当前无仓位，遇到卖出信号，开空仓
        elif signal == -1 and position == 0:
            # 计算仓位大小
            trade_size = capital * position_size
            # 计算手续费
            fee = trade_size * fee_rate
            # 计算扣除手续费后的金额
            amount_after_fees = trade_size - fee
            # 计算卖空股数
            shares_shorted = amount_after_fees / curr_price
            # 更新仓位
            position = -1
            entry_price = curr_price
            capital -= fee  # 只扣除手续费
            holdings = -shares_shorted * curr_price  # 负值表示空头持仓
            hold_periods = 0  # 重置持仓周期计数器

        # 情况3: 当前持有多仓，遇到卖出信号或强制平仓，平多仓
        elif (signal == -1 or force_close) and position == 1:
            # 计算扣除手续费前的价值
            value_before_fees = holdings * curr_price / entry_price
            # 计算手续费
            fee = value_before_fees * fee_rate
            # 更新资金
            capital += value_before_fees - fee

            # 如果有卖出信号，则直接开空仓
            if signal == -1 and not force_close:
                # 计算新仓位大小
                trade_size = capital * position_size
                # 计算手续费
                fee = trade_size * fee_rate
                # 计算扣除手续费后的金额
                amount_after_fees = trade_size - fee
                # 计算卖空股数
                shares_shorted = amount_after_fees / curr_price
                # 更新仓位
                position = -1
                entry_price = curr_price
                capital -= fee  # 只扣除手续费
                holdings = -shares_shorted * curr_price  # 负值表示空头持仓
                hold_periods = 0  # 重置持仓周期计数器
            else:
                # 没有新信号或强制平仓，则仅平仓
                position = 0
                entry_price = 0
                holdings = 0
                hold_periods = 0

        # 情况4: 当前持有空仓，遇到买入信号或强制平仓，平空仓
        elif (signal == 1 or force_close) and position == -1:
            # 计算空仓价值(负值)
            short_value = holdings
            # 计算平仓后的收益(持仓价值减少意味着赚钱)
            short_profit = -short_value * (1 - curr_price / entry_price)
            # 计算手续费
            fee = abs(short_profit) * fee_rate
            # 更新资金
            capital += short_profit - fee

            # 如果有买入信号，则直接开多仓
            if signal == 1 and not force_close:
                # 计算新仓位大小
                trade_size = capital * position_size
                # 计算手续费
                fee = trade_size * fee_rate
                # 计算扣除手续费后的金额
                amount_after_fees = trade_size - fee
                # 计算买入股数
                shares_bought = amount_after_fees / curr_price
                # 更新仓位
                position = 1
                entry_price = curr_price
                capital -= trade_size
                holdings = shares_bought * curr_price
                hold_periods = 0  # 重置持仓周期计数器
            else:
                # 没有新信号或强制平仓，则仅平仓
                position = 0
                entry_price = 0
                holdings = 0
                hold_periods = 0

        # 如果持有仓位，增加持仓周期计数
        if position != 0:
            hold_periods += 1

        # 计算当前持仓的价值
        if position == 1:
            holdings_value = holdings * curr_price / entry_price
        elif position == -1:
            # 空头持仓，计算当前价值
            shares_shorted = abs(holdings) / entry_price
            holdings_value = -shares_shorted * curr_price
        else:
            holdings_value = 0

        # 更新回测DataFrame
        backtest_df.loc[backtest_df.index[i], "position"] = position
        backtest_df.loc[backtest_df.index[i], "entry_price"] = entry_price
        backtest_df.loc[backtest_df.index[i], "capital"] = capital
        backtest_df.loc[backtest_df.index[i], "holdings"] = holdings_value
        backtest_df.loc[backtest_df.index[i], "total_value"] = capital + holdings_value
        backtest_df.loc[backtest_df.index[i], "hold_periods"] = hold_periods

    return backtest_df


def analyze_results(backtest_df, symbol):
    """
    分析回测结果并计算各种性能指标

    参数:
        backtest_df: 回测结果DataFrame
        symbol: 交易对符号

    返回:
        dict: 包含性能指标的字典
    """
    # 提取结果数据
    initial_capital = backtest_df["total_value"].iloc[0]
    final_capital = backtest_df["total_value"].iloc[-1]
    total_return = final_capital / initial_capital - 1

    # 计算每日回报率
    backtest_df["daily_return"] = backtest_df["total_value"].pct_change()

    # 计算累计回报率
    backtest_df["cumulative_return"] = (1 + backtest_df["daily_return"]).cumprod() - 1

    # 计算最大回撤
    backtest_df["peak"] = backtest_df["total_value"].cummax()
    backtest_df["drawdown"] = (
        backtest_df["total_value"] - backtest_df["peak"]
    ) / backtest_df["peak"]
    max_drawdown = backtest_df["drawdown"].min()

    # 计算年化回报率
    trading_days = len(backtest_df)
    annual_return = ((1 + total_return) ** (252 / trading_days)) - 1

    # 计算夏普比率
    daily_returns = backtest_df["daily_return"].dropna()
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / (daily_returns.std() + 1e-10))

    # 计算卡尔马比率
    calmar_ratio = abs(annual_return / (max_drawdown + 1e-10))

    # 计算交易统计
    backtest_df["trade_entry"] = backtest_df["position"].diff() != 0
    num_trades = backtest_df["trade_entry"].sum()

    # 分析多头和空头交易
    long_entries = (backtest_df["position"] == 1) & (
        backtest_df["position"].shift(1) != 1
    )
    short_entries = (backtest_df["position"] == -1) & (
        backtest_df["position"].shift(1) != -1
    )
    long_exits = (backtest_df["position"] != 1) & (
        backtest_df["position"].shift(1) == 1
    )
    short_exits = (backtest_df["position"] != -1) & (
        backtest_df["position"].shift(1) == -1
    )

    num_long_trades = long_entries.sum()
    num_short_trades = short_entries.sum()

    # 计算胜率
    if num_trades > 0:
        # 分析多头和空头交易结果
        long_trade_returns = []
        short_trade_returns = []

        for i in range(len(backtest_df)):
            if long_entries.iloc[i]:
                entry_price = backtest_df["close"].iloc[i]
                entry_total_value = backtest_df["total_value"].iloc[i]
                # 查找下一个平仓点
                exit_idx = next(
                    (j for j in range(i + 1, len(backtest_df)) if long_exits.iloc[j]),
                    None,
                )
                if exit_idx:
                    exit_total_value = backtest_df["total_value"].iloc[exit_idx]
                    trade_return = exit_total_value / entry_total_value - 1
                    long_trade_returns.append(trade_return)

            if short_entries.iloc[i]:
                entry_price = backtest_df["close"].iloc[i]
                entry_total_value = backtest_df["total_value"].iloc[i]
                # 查找下一个平仓点
                exit_idx = next(
                    (j for j in range(i + 1, len(backtest_df)) if short_exits.iloc[j]),
                    None,
                )
                if exit_idx:
                    exit_total_value = backtest_df["total_value"].iloc[exit_idx]
                    trade_return = exit_total_value / entry_total_value - 1
                    short_trade_returns.append(trade_return)

        # 计算胜率
        long_win_rate = sum(1 for r in long_trade_returns if r > 0) / (
            len(long_trade_returns) or 1
        )
        short_win_rate = sum(1 for r in short_trade_returns if r > 0) / (
            len(short_trade_returns) or 1
        )
        total_win_rate = (
            sum(1 for r in long_trade_returns if r > 0)
            + sum(1 for r in short_trade_returns if r > 0)
        ) / (len(long_trade_returns) + len(short_trade_returns) or 1)
    else:
        long_win_rate = 0
        short_win_rate = 0
        total_win_rate = 0

    # 收集结果
    results = {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "num_trades": num_trades,
        "num_long_trades": num_long_trades,
        "num_short_trades": num_short_trades,
        "win_rate": total_win_rate,
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
    }

    # 打印结果摘要
    print(f"\n====== {symbol} 回测结果 ======")
    print(f"初始资金: ${initial_capital:.2f}")
    print(f"最终资金: ${final_capital:.2f}")
    print(f"总回报率: {total_return:.2%}")
    print(f"年化回报率: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"卡尔马比率: {calmar_ratio:.2f}")
    print(f"交易次数: {num_trades}")
    print(f"  多头交易: {num_long_trades}")
    print(f"  空头交易: {num_short_trades}")
    print(f"总胜率: {total_win_rate:.2%}")
    print(f"  多头胜率: {long_win_rate:.2%}")
    print(f"  空头胜率: {short_win_rate:.2%}")

    return results, backtest_df


def plot_results(backtest_df, symbol, save_dir="backtest_results"):
    """
    绘制回测结果图表

    参数:
        backtest_df: 回测结果DataFrame
        symbol: 交易对符号
        save_dir: 保存图表的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 设置图表风格
    plt.style.use("dark_background")

    # 1. 绘制价格和交易信号
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制价格
    ax.plot(
        backtest_df.index,
        backtest_df["close"],
        color="white",
        linewidth=1,
        label="price",
    )

    # 绘制买入点
    buy_signals = backtest_df[
        (backtest_df["position"] == 1) & (backtest_df["position"].shift(1) != 1)
    ]
    ax.scatter(
        buy_signals.index,
        buy_signals["close"],
        color="green",
        marker="^",
        s=100,
        label="buy",
    )

    # 绘制卖出点
    sell_signals = backtest_df[
        (backtest_df["position"] == -1) & (backtest_df["position"].shift(1) != -1)
    ]
    ax.scatter(
        sell_signals.index,
        sell_signals["close"],
        color="red",
        marker="v",
        s=100,
        label="sell",
    )

    # 绘制平仓点
    close_long = backtest_df[
        (backtest_df["position"] != 1) & (backtest_df["position"].shift(1) == 1)
    ]
    close_short = backtest_df[
        (backtest_df["position"] != -1) & (backtest_df["position"].shift(1) == -1)
    ]
    ax.scatter(
        close_long.index,
        close_long["close"],
        color="orange",
        marker="x",
        s=80,
        label="close long",
    )
    ax.scatter(
        close_short.index,
        close_short["close"],
        color="orange",
        marker="x",
        s=80,
        label="close short",
    )

    # 设置标题和标签
    ax.set_title(f"{symbol} price and signals", fontsize=16)
    ax.set_xlabel("date", fontsize=12)
    ax.set_ylabel("price", fontsize=12)
    ax.legend()
    # ax.grid(alpha=0.3)

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{symbol}_price_signals.png"))
    plt.close()

    # 2. 绘制资金曲线
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制资金曲线
    ax.plot(
        backtest_df.index,
        backtest_df["total_value"],
        color="cyan",
        linewidth=2,
        label="equity curve",
    )

    # 设置标题和标签
    ax.set_title(f"{symbol} equity curve", fontsize=16)
    ax.set_xlabel("date", fontsize=12)
    ax.set_ylabel("equity", fontsize=12)
    ax.legend()
    # ax.grid(alpha=0.3)

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{symbol}_equity_curve.png"))
    plt.close()

    # 3. 绘制回撤
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制回撤
    ax.fill_between(
        backtest_df.index,
        backtest_df["drawdown"] * 100,
        0,
        color="red",
        alpha=0.3,
        label="drawdown %",
    )
    ax.plot(backtest_df.index, backtest_df["drawdown"] * 100, color="red", linewidth=1)

    # 设置标题和标签
    ax.set_title(f"{symbol} drawdown (%)", fontsize=16)
    ax.set_xlabel("date", fontsize=12)
    ax.set_ylabel("drawdown (%)", fontsize=12)
    # ax.grid(alpha=0.3)

    # 反转Y轴，使回撤向下显示
    ax.invert_yaxis()

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{symbol}_drawdown.png"))
    plt.close()

    # 4. 绘制信号分布
    if "probability" in backtest_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))

        # 筛选有信号的数据
        signal_data = backtest_df[backtest_df["signal"] != 0]
        if not signal_data.empty:
            # 绘制概率分布
            ax.scatter(
                signal_data.index,
                signal_data["probability"],
                c=signal_data["signal"].map({1: "green", -1: "red"}),
                alpha=0.7,
                s=50,
            )

            # 设置标题和标签
            ax.set_title(f"{symbol} signal probabilities", fontsize=16)
            ax.set_xlabel("date", fontsize=12)
            ax.set_ylabel("signal probability", fontsize=12)
            # ax.grid(alpha=0.3)

            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{symbol}_signal_probabilities.png"))
            plt.close()


def process_csv_file(
    csv_file,
    model,
    scaler,
    interval="4h",
    lookback=100,
    step=1,
    probability_threshold=0.7,
):
    """
    处理单个CSV文件进行回测

    参数:
        csv_file: CSV文件路径
        model: CNN模型
        scaler: 用于标准化的scaler
        interval: 时间间隔 (1h, 4h)
        lookback: 用于预测的历史数据长度
        step: 预测步长
        probability_threshold: 概率阈值，只有当预测概率大于或等于此阈值时才生成信号

    返回:
        结果字典和回测DataFrame
    """
    try:
        # 从文件名中提取币种对
        file_name = os.path.basename(csv_file)
        symbol = file_name.split("_")[0]

        print(f"处理 {symbol}...")

        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 清洗数据
        df = clean_data(df)

        # 如果数据量不足，则跳过
        if len(df) < lookback:
            print(f"数据量不足: {len(df)} < {lookback}")
            return None, None

        # 生成CNN预测信号
        signal_df = generate_signals(
            df, model, scaler, lookback, step, probability_threshold
        )

        # 回测
        backtest_df = backtest(
            signal_df,
            initial_capital=10000.0,
            position_size=1.0,
            fee_rate=0.0005,
            max_hold_periods=1,
        )

        # 分析结果
        results, backtest_df = analyze_results(backtest_df, symbol)

        # 绘制结果图表
        plot_results(backtest_df, symbol)

        return results, backtest_df

    except Exception as e:
        print(f"处理 {csv_file} 时出错: {e}")
        return None, None


def main():
    import time

    # 记录开始时间
    start_time = time.time()

    # 模型和scaler路径
    model_path = "run/cnn_model_4h.pth"  # 根据实际路径修改
    scaler_path = "run/scaler.pkl"  # 根据实际路径修改

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件未找到: {model_path}")
        print("请提供正确的模型文件路径")
        return

    if not os.path.exists(scaler_path):
        print(f"Scaler文件未找到: {scaler_path}")
        print("请提供正确的scaler文件路径")
        return

    # 加载模型和scaler
    model = load_model(model_path, model_type="4h", device=device)
    scaler = load_scaler(scaler_path)

    # 要处理的CSV文件目录
    csv_dir = "merged_csv"  # 根据实际路径修改

    if not os.path.exists(csv_dir):
        print(f"CSV目录未找到: {csv_dir}")
        print("请提供包含历史数据CSV文件的目录")
        return

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(csv_dir, "*_4h_*.csv"))

    if not csv_files:
        print(f"在 {csv_dir} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 回测参数
    interval = "4h"  # 或 "1h"，取决于模型类型
    lookback = 100  # 历史数据长度
    step = 1  # 预测步长
    probability_threshold = 0.9  # 概率阈值

    # 处理所有文件
    all_results = []

    for csv_file in tqdm(csv_files, desc="处理文件"):
        results, _ = process_csv_file(
            csv_file, model, scaler, interval, lookback, step, probability_threshold
        )
        if results:
            all_results.append(results)

    # 创建结果目录
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(
            os.path.join(
                results_dir, f"backtest_summary_threshold_{probability_threshold}.csv"
            ),
            index=False,
        )

        print("\n=== 回测汇总 ===")
        print(f"概率阈值: {probability_threshold}")
        print(f"总交易对数: {len(summary_df)}")
        print(f"平均总回报率: {summary_df['total_return'].mean():.2%}")
        print(f"平均年化回报率: {summary_df['annual_return'].mean():.2%}")
        print(f"平均最大回撤: {summary_df['max_drawdown'].mean():.2%}")
        print(f"平均夏普比率: {summary_df['sharpe_ratio'].mean():.2f}")
        print(f"平均胜率: {summary_df['win_rate'].mean():.2%}")

    # 记录结束时间和总耗时
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n总耗时: {duration:.2f} 秒")
    print(f"结果已保存到 {results_dir} 目录")


if __name__ == "__main__":
    main()
