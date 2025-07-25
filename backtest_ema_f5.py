import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import concurrent.futures
import time
from tqdm import tqdm
import multiprocessing
import psutil
import warnings
from config_backtest import (
    STRATEGY_CONFIG,
    BACKTEST_CONFIG,
    DATA_CONFIG,
    RESULTS_CONFIG,
    MULTIPROCESSING_CONFIG,
)

# Ignore warnings to reduce output noise
warnings.filterwarnings("ignore")


# Create a Strategy class to decouple strategy parameters
class F5Strategy:
    def __init__(self, ema_window=None, sma_window=None):
        # Use config values or provided values
        self.ema_window = (
            ema_window if ema_window is not None else STRATEGY_CONFIG["ema_window"]
        )
        self.sma_window = (
            sma_window if sma_window is not None else STRATEGY_CONFIG["sma_window"]
        )

    def calculate_ema(self, df):
        """Calculate Exponential Moving Average"""
        return df["close"].ewm(span=self.ema_window, adjust=False).mean()

    def calculate_sma(self, df):
        """Calculate Simple Moving Average"""
        return df["close"].rolling(window=self.sma_window).mean()

    def generate_signals(self, df):
        """Generate trading signals based on EMA/SMA crossover"""
        # Make a copy to avoid modifying the original
        strategy_df = df.copy()

        # Calculate indicators
        # strategy_df["ema"] = self.calculate_ema(strategy_df)
        # strategy_df["sma"] = self.calculate_sma(strategy_df)
        strategy_df["sma_60"] = sma(strategy_df, window=60)
        # strategy_df["kdj_k"], strategy_df["kdj_d"], strategy_df["kdj_j"] = kdj(
        #     strategy_df, window=9, k_s=3, d_s=3
        # )
        strategy_df["macd"], strategy_df["dif"], strategy_df["dea"] = macd(
            strategy_df, fast_period=6, slow_period=12, signal_period=9
        )
        # strategy_df["bb_upper"], strategy_df["bb_lower"] = bb(
        #     strategy_df, window=20, std_dev=0.8
        # )
        strategy_df["rsi"] = rsi(strategy_df, window=6)
        strategy_df["stochrsi"], strategy_df["mastochrsi"] = stochrsi(
            strategy_df, rsi_col="rsi", stoch_window=6, k_window=3, d_window=3
        )
        # Initialize signal column
        strategy_df["signal"] = 0

        # Buy signals (1)
        """
        buy_condition = (
            (strategy_df["ema"].shift(1) < strategy_df["sma"].shift(1))
            & (strategy_df["ema"] > strategy_df["sma"])
            & (strategy_df["ema"] > strategy_df["ema"].shift(1))
            & (strategy_df["sma"] > strategy_df["sma"].shift(1))
            & (strategy_df["close"] - strategy_df["open"] > 0)
            & (strategy_df["close"].shift(1) - strategy_df["open"].shift(1) < 0)
            & (strategy_df["bb_upper"] > strategy_df["bb_upper"].shift(1))
            & (strategy_df["bb_lower"] > strategy_df["bb_lower"].shift(1))
            & (strategy_df["sma"] > strategy_df["bb_upper"])
            & (strategy_df["macd"] > 0)
        )

        # Sell signals (-1)
        sell_condition = (
            (strategy_df["ema"].shift(1) > strategy_df["sma"].shift(1))
            & (strategy_df["ema"] < strategy_df["sma"])
            & (strategy_df["ema"] < strategy_df["ema"].shift(1))
            & (strategy_df["sma"] < strategy_df["sma"].shift(1))
            & (strategy_df["close"] - strategy_df["open"] < 0)
            & (strategy_df["close"].shift(1) - strategy_df["open"].shift(1) > 0)
            & (strategy_df["bb_upper"] < strategy_df["bb_upper"].shift(1))
            & (strategy_df["bb_lower"] < strategy_df["bb_lower"].shift(1))
            & (strategy_df["sma"] < strategy_df["bb_lower"])
            & (strategy_df["macd"] < 0)
        )
        """

        buy_condition = (
            (strategy_df["sma_60"] > strategy_df["sma_60"].shift(1))
            & (strategy_df["stochrsi"].shift(1) < strategy_df["mastochrsi"].shift(1))
            & (strategy_df["stochrsi"] > strategy_df["stochrsi"].shift(1))
            & (strategy_df["macd"] > 0)
            & (strategy_df["dea"] > 0)
        )

        sell_condition = (
            (strategy_df["sma_60"] < strategy_df["sma_60"].shift(1))
            & (strategy_df["stochrsi"].shift(1) > strategy_df["mastochrsi"].shift(1))
            & (strategy_df["stochrsi"] < strategy_df["stochrsi"].shift(1))
            & (strategy_df["macd"] < 0)
            & (strategy_df["dea"] < 0)
        )

        # Set signals
        strategy_df.loc[buy_condition, "signal"] = 1
        strategy_df.loc[sell_condition, "signal"] = -1

        # Filter out signals that occur within 15 periods of a previous signal of the same direction
        # Make a copy of the original signals
        original_signals = strategy_df["signal"].copy()

        # Iterate through the dataframe to check previous signals
        for i in range(len(strategy_df)):
            if (
                strategy_df["signal"].iloc[i] != 0
            ):  # If there's a signal at this position
                signal_direction = strategy_df["signal"].iloc[
                    i
                ]  # Get signal direction (1 or -1)

                # Define the lookback period (min of current index and 15)
                lookback = min(i, 15)

                # Check if there's a signal of the same direction in the previous 15 periods
                if lookback > 0:
                    previous_signals = original_signals.iloc[i - lookback : i]
                    if any(previous_signals == signal_direction):
                        # If there was a signal of the same direction in the lookback period,
                        # cancel this signal by setting it to 0
                        strategy_df["signal"].iloc[i] = 0

        return strategy_df


# Legacy indicator functions for backward compatibility
def sma(df, window=None):
    if window is None:
        window = STRATEGY_CONFIG["sma_window"]
    return df["close"].rolling(window=window).mean()


def ema(df, window=None):
    if window is None:
        window = STRATEGY_CONFIG["ema_window"]
    return df["close"].ewm(span=window, adjust=False).mean()


def macd(df, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line - signal_line, macd_line, signal_line


def kdj(df, window=9, k_s=3, d_s=3):
    high = df["high"].rolling(window=window, min_periods=1).max()
    low = df["low"].rolling(window=window, min_periods=1).min()
    rsv = (df["close"] - low) / (high - low) * 100
    k = rsv.ewm(alpha=1 / k_s, adjust=False).mean()
    d = k.ewm(alpha=1 / d_s, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def bb(df, window=20, std_dev=2):
    """Calculate Bollinger Bands"""
    moving_avg = df["close"].rolling(window=window).mean()
    std_dev = df["close"].rolling(window=window).std()
    upper_band = moving_avg + std_dev * std_dev
    lower_band = moving_avg - std_dev * std_dev
    return upper_band, lower_band


def rsi(df, window=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    first_gain = gain.rolling(window).mean().shift(1)
    first_loss = loss.rolling(window).mean().shift(1)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean().combine_first(first_gain)
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean().combine_first(first_loss)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def stochrsi(
    df,
    rsi_col="rsi",
    stoch_window=14,
    k_window=3,
    d_window=3,
):

    rsi = df[rsi_col]

    rsi_min = rsi.rolling(window=stoch_window).min()
    rsi_max = rsi.rolling(window=stoch_window).max()

    stoch = (rsi - rsi_min) / (rsi_max - rsi_min)

    stochrsi = stoch.rolling(window=k_window).mean() * 100
    mastochrsi = stochrsi.rolling(window=d_window).mean()

    return stochrsi, mastochrsi


def clean_data(df):
    """Clean and preprocess the data from CSV files"""
    # Convert timestamp to datetime
    df["open_time"] = pd.to_datetime(df["open_time"])

    df["open_time"] = df["open_time"]

    # Set index to datetime
    df = df.set_index("open_time")

    df.drop(
        columns=[
            "volume",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ],
        inplace=True,
        errors="ignore",
    )

    # Ensure numeric columns are float
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df


# Legacy strategy function for backward compatibility
def f5_strategy(df):
    strategy = F5Strategy()
    return strategy.generate_signals(df)


def backtest(
    df, initial_capital=None, position_size=None, fee_rate=None, max_hold_periods=None
):
    # Use config values if parameters are not provided
    if initial_capital is None:
        initial_capital = BACKTEST_CONFIG["initial_capital"]
    if position_size is None:
        position_size = BACKTEST_CONFIG["position_size"]
    if fee_rate is None:
        fee_rate = BACKTEST_CONFIG["fee_rate"]
    if max_hold_periods is None:
        max_hold_periods = BACKTEST_CONFIG["max_hold_periods"]

    # Create a copy of the DataFrame to avoid modifying the original
    backtest_df = df.copy()

    # Initialize columns for backtesting
    backtest_df["position"] = 0  # 0 = no position, 1 = long, -1 = short
    backtest_df["entry_price"] = 0.0  # Entry price for current position
    backtest_df["capital"] = initial_capital  # Current available capital
    backtest_df["margin"] = 0.0  # Amount of capital allocated as margin
    backtest_df["holdings"] = (
        0.0  # Current value of holdings (can be positive or negative)
    )
    backtest_df["total_value"] = initial_capital  # Capital + holdings
    backtest_df["hold_periods"] = 0  # Number of periods a position has been held
    backtest_df["forced_close"] = (
        0  # Indicator for forced close due to max hold periods
    )

    # Pre-calculate arrays for better performance
    close_prices = backtest_df["close"].values
    signals = backtest_df["signal"].values

    # Initialize variables
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    capital = initial_capital
    margin = 0.0  # Amount allocated as margin
    holdings = 0.0
    hold_periods = 0

    # Arrays to store results
    positions = np.zeros(len(backtest_df))
    entry_prices = np.zeros(len(backtest_df))
    capitals = np.zeros(len(backtest_df))
    margins = np.zeros(len(backtest_df))
    holdings_array = np.zeros(len(backtest_df))
    total_values = np.zeros(len(backtest_df))
    hold_periods_array = np.zeros(len(backtest_df))
    forced_closes = np.zeros(len(backtest_df))

    # Set initial values
    positions[0] = 0
    entry_prices[0] = 0
    capitals[0] = initial_capital
    margins[0] = 0
    holdings_array[0] = 0
    total_values[0] = initial_capital
    hold_periods_array[0] = 0
    forced_closes[0] = 0

    # Loop through the data to simulate trading (start from index 1)
    for i in range(1, len(backtest_df)):
        curr_price = close_prices[i]  # 当前K线收盘价（用于平仓）
        prev_price = close_prices[i - 1]  # 前一K线收盘价（用于开仓）
        signal = signals[i]

        # 强制平仓标志
        force_close = False
        if position != 0 and hold_periods >= max_hold_periods:
            force_close = True
            forced_closes[i] = 1

        # 处理信号和强制平仓
        # 情况1: 当前无仓位，遇到买入信号，开多仓
        if signal == 1 and position == 0:
            # Calculate margin size (amount to allocate to the position)
            margin = capital * position_size
            # Calculate fees
            fee = margin * fee_rate
            # Reduce available capital
            capital -= margin
            capital -= fee  # Deduct fee from capital
            # Update position
            position = 1
            entry_price = prev_price  # 使用前一K线收盘价为入场价格
            # Calculate current holdings value
            holdings = margin * (curr_price / entry_price)
            hold_periods = 0  # 重置持仓周期计数器

        # 情况2: 当前无仓位，遇到卖出信号，开空仓
        elif signal == -1 and position == 0:
            # Calculate margin size (amount to allocate to the position)
            margin = capital * position_size
            # Calculate fees
            fee = margin * fee_rate
            # Reduce available capital
            capital -= margin
            capital -= fee  # Deduct fee from capital
            # Update position
            position = -1
            entry_price = prev_price  # 使用前一K线收盘价为入场价格
            # Calculate current holdings value (negative for short)
            holdings = -margin * (entry_price / curr_price)
            hold_periods = 0  # 重置持仓周期计数器

        # 情况3: 当前持有多仓，遇到卖出信号或强制平仓，平多仓
        elif (signal == -1 or force_close) and position == 1:
            # Calculate the current value of holdings
            current_value = margin * (curr_price / entry_price)
            # Calculate fees for closing (based on current value)
            fee = current_value * fee_rate
            # Return margin plus/minus profit/loss and minus fees to capital
            capital += current_value - fee

            # Reset position
            position = 0
            entry_price = 0
            margin = 0.0
            holdings = 0.0
            hold_periods = 0  # 重置持仓周期计数器

        # 情况4: 当前持有空仓，遇到买入信号或强制平仓，平空仓
        elif (signal == 1 or force_close) and position == -1:
            # Calculate the current value of short position
            # For shorts: positive return when price falls
            current_value = margin * (entry_price / curr_price)
            # Calculate fees for closing (based on current value)
            fee = current_value * fee_rate
            # Return margin plus/minus profit/loss and minus fees to capital
            capital += current_value - fee

            # Reset position
            position = 0
            entry_price = 0
            margin = 0.0
            holdings = 0.0
            hold_periods = 0  # 重置持仓周期计数器

        # 更新持仓价值
        if position == 1:
            # 多头持仓价值随价格变化
            holdings = margin * (curr_price / entry_price)
            hold_periods += 1
        elif position == -1:
            # 空头持仓价值随价格变化
            holdings = -margin * (entry_price / curr_price)
            hold_periods += 1

        # Store values in arrays
        positions[i] = position
        entry_prices[i] = entry_price
        capitals[i] = capital
        margins[i] = margin
        holdings_array[i] = holdings

        # Calculate total value
        total_values[i] = capital + holdings
        hold_periods_array[i] = hold_periods

    # Update the DataFrame with the calculated arrays
    backtest_df["position"] = positions
    backtest_df["entry_price"] = entry_prices
    backtest_df["capital"] = capitals
    backtest_df["margin"] = margins
    backtest_df["holdings"] = holdings_array
    backtest_df["total_value"] = total_values
    backtest_df["hold_periods"] = hold_periods_array
    backtest_df["forced_close"] = forced_closes

    # Calculate returns and metrics
    backtest_df["return"] = (
        backtest_df["total_value"] / backtest_df["total_value"].iloc[0] - 1
    )
    backtest_df["benchmark_return"] = (
        backtest_df["close"] / backtest_df["close"].iloc[0] - 1
    )

    # Calculate drawdown more efficiently
    backtest_df["cummax"] = np.maximum.accumulate(backtest_df["total_value"].values)
    backtest_df["drawdown"] = (
        backtest_df["total_value"] / backtest_df["cummax"] - 1
    ) * 100

    return backtest_df


def analyze_results(backtest_df, symbol):
    """Calculate and print performance metrics"""
    # Get final values
    initial_capital = backtest_df["total_value"].iloc[0]
    final_value = backtest_df["total_value"].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Buy and hold return
    buy_hold_return = (
        backtest_df["close"].iloc[-1] / backtest_df["close"].iloc[0] - 1
    ) * 100

    # 计算仓位变化
    backtest_df["position_change"] = backtest_df["position"].diff()

    # 统计交易次数
    # 多头开仓次数
    long_entries = len(
        backtest_df[
            (backtest_df["position_change"] == 1) & (backtest_df["position"] == 1)
        ]
    )
    # 多头平仓次数
    long_exits = len(
        backtest_df[
            (backtest_df["position_change"] < 0)
            & (backtest_df["position"].shift(1) == 1)
        ]
    )
    # 空头开仓次数
    short_entries = len(
        backtest_df[
            (backtest_df["position_change"] == -1) & (backtest_df["position"] == -1)
        ]
    )
    # 空头平仓次数
    short_exits = len(
        backtest_df[
            (backtest_df["position_change"] > 0)
            & (backtest_df["position"].shift(1) == -1)
        ]
    )

    # 计算强制平仓次数
    forced_closes = 0
    if "forced_close" in backtest_df.columns:
        forced_closes = backtest_df["forced_close"].sum()

    # 计算多头和空头的强制平仓次数
    forced_long_closes = len(
        backtest_df[
            (backtest_df["forced_close"] == 1) & (backtest_df["position"].shift(1) == 1)
        ]
    )
    forced_short_closes = len(
        backtest_df[
            (backtest_df["forced_close"] == 1)
            & (backtest_df["position"].shift(1) == -1)
        ]
    )

    # 总交易次数 = 多头交易 + 空头交易
    total_trades = long_entries + short_entries

    # 分析多头和空头交易的表现
    # 提取每次交易的收益率
    trades_data = []
    current_position = 0
    entry_price = 0
    entry_date = None

    for i in range(1, len(backtest_df)):
        # 检测仓位变化
        if backtest_df["position"].iloc[i] != current_position:
            # 如果之前有持仓，现在平仓或换仓，则记录交易结果
            if current_position != 0:
                exit_price = backtest_df["close"].iloc[i]
                exit_date = backtest_df.index[i]

                # 计算收益率
                if current_position == 1:  # 多头
                    profit_pct = (exit_price / entry_price - 1) * 100
                    trade_type = "LONG"
                else:  # 空头
                    profit_pct = (1 - exit_price / entry_price) * 100
                    trade_type = "SHORT"

                # 判断是否为强制平仓
                forced = backtest_df["forced_close"].iloc[i] == 1

                trades_data.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "trade_type": trade_type,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "profit_pct": profit_pct,
                        "forced": forced,
                        "capital_after_trade": backtest_df["capital"].iloc[i],
                    }
                )

            # 更新当前仓位，如果是开仓则记录入场价格
            current_position = backtest_df["position"].iloc[i]
            if current_position != 0:
                entry_price = backtest_df["close"].iloc[i]
                entry_date = backtest_df.index[i]

    # 创建交易DataFrame
    if trades_data:
        trades_df = pd.DataFrame(trades_data)

        # 计算多头和空头的胜率和平均收益
        if len(trades_df[trades_df["trade_type"] == "LONG"]) > 0:
            long_win_rate = (
                len(
                    trades_df[
                        (trades_df["trade_type"] == "LONG")
                        & (trades_df["profit_pct"] > 0)
                    ]
                )
                / len(trades_df[trades_df["trade_type"] == "LONG"])
                * 100
            )
            long_avg_profit = trades_df[trades_df["trade_type"] == "LONG"][
                "profit_pct"
            ].mean()
            long_max_profit = trades_df[trades_df["trade_type"] == "LONG"][
                "profit_pct"
            ].max()
            long_max_loss = trades_df[trades_df["trade_type"] == "LONG"][
                "profit_pct"
            ].min()
        else:
            long_win_rate = 0
            long_avg_profit = 0
            long_max_profit = 0
            long_max_loss = 0

        if len(trades_df[trades_df["trade_type"] == "SHORT"]) > 0:
            short_win_rate = (
                len(
                    trades_df[
                        (trades_df["trade_type"] == "SHORT")
                        & (trades_df["profit_pct"] > 0)
                    ]
                )
                / len(trades_df[trades_df["trade_type"] == "SHORT"])
                * 100
            )
            short_avg_profit = trades_df[trades_df["trade_type"] == "SHORT"][
                "profit_pct"
            ].mean()
            short_max_profit = trades_df[trades_df["trade_type"] == "SHORT"][
                "profit_pct"
            ].max()
            short_max_loss = trades_df[trades_df["trade_type"] == "SHORT"][
                "profit_pct"
            ].min()
        else:
            short_win_rate = 0
            short_avg_profit = 0
            short_max_profit = 0
            short_max_loss = 0

        # 总体胜率
        overall_win_rate = (
            len(trades_df[trades_df["profit_pct"] > 0]) / len(trades_df) * 100
        )
    else:
        long_win_rate = 0
        long_avg_profit = 0
        long_max_profit = 0
        long_max_loss = 0
        short_win_rate = 0
        short_avg_profit = 0
        short_max_profit = 0
        short_max_loss = 0
        overall_win_rate = 0

    # Calculate daily returns for annualized metrics
    backtest_df["daily_return"] = backtest_df["total_value"].pct_change()

    # Calculate annualized return (assuming 365 trading days in a year)
    days = (backtest_df.index[-1] - backtest_df.index[0]).days
    years = days / 365
    annualized_return = (
        ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    )

    # Calculate max drawdown more efficiently
    backtest_df["cummax"] = backtest_df["total_value"].cummax()
    backtest_df["drawdown"] = (
        backtest_df["total_value"] / backtest_df["cummax"] - 1
    ) * 100
    max_drawdown = backtest_df["drawdown"].min()

    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    risk_free_rate = 0
    daily_std = backtest_df["daily_return"].std()
    # FIX: Properly handle zero standard deviation
    if daily_std is None or np.isnan(daily_std) or daily_std == 0:
        sharpe_ratio = 0
    else:
        daily_mean = backtest_df["daily_return"].mean()
        if np.isnan(daily_mean):
            sharpe_ratio = 0
        else:
            sharpe_ratio = (daily_mean - risk_free_rate) / daily_std * (365**0.5)

    # 准备要输出的性能摘要文本
    summary_text = f"\n----- Performance Summary for {symbol} -----\n"
    summary_text += f"Initial Capital: ${initial_capital:.2f}\n"
    summary_text += f"Final Value: ${final_value:.2f}\n"
    summary_text += f"Total Return: {total_return:.2f}%\n"
    summary_text += f"Buy & Hold Return: {buy_hold_return:.2f}%\n"
    summary_text += f"Outperformance: {total_return - buy_hold_return:.2f}%\n"
    summary_text += f"Annualized Return: {annualized_return:.2f}%\n"
    summary_text += f"Maximum Drawdown: {max_drawdown:.2f}%\n"
    summary_text += f"Sharpe Ratio: {sharpe_ratio:.2f}\n\n"

    summary_text += f"--- Trading Statistics ---\n"
    summary_text += f"Total Trades: {total_trades}\n"
    summary_text += f"Overall Win Rate: {overall_win_rate:.2f}%\n\n"

    summary_text += f"Long Trades: {long_entries}\n"
    summary_text += f"Long Win Rate: {long_win_rate:.2f}%\n"
    summary_text += f"Long Avg Profit: {long_avg_profit:.2f}%\n"
    summary_text += f"Long Max Profit: {long_max_profit:.2f}%\n"
    summary_text += f"Long Max Loss: {long_max_loss:.2f}%\n"
    summary_text += f"Long Forced Closes: {forced_long_closes}\n\n"

    summary_text += f"Short Trades: {short_entries}\n"
    summary_text += f"Short Win Rate: {short_win_rate:.2f}%\n"
    summary_text += f"Short Avg Profit: {short_avg_profit:.2f}%\n"
    summary_text += f"Short Max Profit: {short_max_profit:.2f}%\n"
    summary_text += f"Short Max Loss: {short_max_loss:.2f}%\n"
    summary_text += f"Short Forced Closes: {forced_short_closes}\n"

    # 打印到控制台
    print(summary_text)

    # 确保目录存在
    log_dir = RESULTS_CONFIG["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # 保存到文本文件
    with open(f"{log_dir}/{symbol}_performance.txt", "w") as f:
        f.write(summary_text)

        # 若有交易记录，添加详细交易记录
        if trades_data:
            f.write("\n\n--- Detailed Trade Records ---\n")
            f.write(trades_df.to_string())

    # 同时将所有结果追加到汇总日志文件
    with open(f"{log_dir}/all_performances.txt", "a") as f:
        f.write(summary_text + "\n" + "-" * 50 + "\n")

    return {
        "symbol": symbol,
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "outperformance": total_return - buy_hold_return,
        "num_trades": total_trades,
        "overall_win_rate": overall_win_rate,
        "long_trades": long_entries,
        "long_win_rate": long_win_rate,
        "long_avg_profit": long_avg_profit,
        "short_trades": short_entries,
        "short_win_rate": short_win_rate,
        "short_avg_profit": short_avg_profit,
        "forced_closes": forced_closes,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }


def plot_results(backtest_df, symbol):
    """Plot the results of the backtest"""
    plt.figure(figsize=(14, 10))

    # Plot 1: Price with buy/sell signals
    plt.subplot(3, 1, 1)
    plt.plot(backtest_df.index, backtest_df["close"], label="Price", alpha=0.5)

    # 计算position列的差分用于其他功能
    backtest_df["position_change"] = backtest_df["position"].diff()

    plt.title(f"{symbol} Price with EMA-SMA Strategy")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(False)

    # Plot 2: Strategy Performance vs Buy & Hold
    plt.subplot(3, 1, 2)
    plt.plot(
        backtest_df.index,
        backtest_df["return"] * 100,
        label="Strategy Return (%)",
        color="blue",
    )
    plt.plot(
        backtest_df.index,
        backtest_df["benchmark_return"] * 100,
        label="Buy & Hold Return (%)",
        color="gray",
        alpha=0.5,
    )

    plt.title(f"{symbol} Strategy Performance vs Buy & Hold")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(False)

    # Plot 3: Drawdowns - Modified to be more intuitive
    plt.subplot(3, 1, 3)
    # Convert drawdown to positive values for better visualization
    positive_drawdown = -backtest_df["drawdown"]  # Make drawdown positive
    plt.fill_between(backtest_df.index, 0, positive_drawdown, color="red", alpha=0.5)
    plt.title(f"{symbol} Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    # Invert y-axis so larger drawdowns appear as deeper dips
    plt.gca().invert_yaxis()
    # Set y-ticks to show drawdown as positive percentages
    max_dd = abs(backtest_df["drawdown"].min())
    if max_dd > 0:
        y_ticks = np.linspace(0, max_dd, 6)
        plt.yticks(y_ticks, [f"{y:.1f}%" for y in y_ticks])
    plt.grid(False)

    plt.tight_layout()

    # 保存图形
    results_dir = RESULTS_CONFIG["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/backtest_results_{symbol}.png", dpi=150)
    plt.close()


def process_single_file(csv_file):
    """
    Process a single cryptocurrency file
    This function handles all steps for one file: reading, cleaning, strategy application,
    backtesting, analysis, and plotting
    """
    try:
        symbol = os.path.basename(csv_file).split("_")[0]
        # Reduce print statements when using multiprocessing to avoid console clutter
        # print(f"Processing {symbol}...")

        # Read CSV file with optimized data types
        dtype_mapping = {
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float32",
            "quote_asset_volume": "float32",
            "number_of_trades": "int32",
            "taker_buy_base_asset_volume": "float32",
            "taker_buy_quote_asset_volume": "float32",
        }

        # Add data validation
        try:
            df = pd.read_csv(csv_file, dtype=dtype_mapping)

            # Validate required columns
            required_columns = ["open_time", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Error: Missing columns in {csv_file}: {missing_columns}")
                return None

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            return None

        # Clean data
        df = clean_data(df)

        # Skip if not enough data
        min_data_length = max(100, STRATEGY_CONFIG["sma_window"] * 2)
        if len(df) < min_data_length:
            # print(f"Skipping {symbol} - not enough data (require {min_data_length} rows)")
            return None

        # Create strategy instance with config parameters
        strategy = F5Strategy(
            ema_window=STRATEGY_CONFIG["ema_window"],
            sma_window=STRATEGY_CONFIG["sma_window"],
        )

        # Apply strategy to generate signals
        df = strategy.generate_signals(df)

        # Use parameters from config file for backtesting
        backtest_df = backtest(
            df,
            initial_capital=BACKTEST_CONFIG["initial_capital"],
            position_size=BACKTEST_CONFIG["position_size"],
            fee_rate=BACKTEST_CONFIG["fee_rate"],
            max_hold_periods=BACKTEST_CONFIG["max_hold_periods"],
        )

        # Analyze results
        result = analyze_results(backtest_df, symbol)

        # Plot results
        plot_results(backtest_df, symbol)

        return result

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None


# 将嵌套函数移到模块级，以解决pickle序列化错误
def process_single_file_with_lock(csv_file, results, file_lock):
    """
    处理单个文件并使用锁保护共享资源

    Args:
        csv_file: 要处理的CSV文件路径
        results: 共享的结果列表
        file_lock: 文件锁对象，用于保护并发写入
    """
    result = process_single_file(csv_file)
    if result:
        with file_lock:
            # 使用特定于符号的文件以减少争用
            symbol = result["symbol"]
            # 这里可以添加其他进程安全的日志记录
        results.append(result)
    return result


def main():
    # Record start time
    start_time = time.time()

    # Create result directories from config
    results_dir = RESULTS_CONFIG["results_dir"]
    log_dir = RESULTS_CONFIG["logs_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Clear summary log file
    with open(os.path.join(log_dir, "all_performances.txt"), "w") as f:
        f.write(f"回测开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n\n")

    # Directory with CSV files from config
    data_dir = DATA_CONFIG["data_dir"]
    file_pattern = DATA_CONFIG["file_pattern"]

    # Get list of CSV files
    csv_files = glob.glob(os.path.join(data_dir, file_pattern))

    # Limit to first 5 files for testing (remove this line to process all files)
    # csv_files = csv_files[:5]

    # Initialize results list manager to share results between processes
    manager = multiprocessing.Manager()
    results = manager.list()

    # Add a lock for file writing to prevent race conditions
    file_lock = manager.Lock()

    # Get CPU info
    cpu_count = psutil.cpu_count(logical=True)
    physical_cpu = psutil.cpu_count(logical=False)

    # Determine optimal number of processes from config
    if MULTIPROCESSING_CONFIG["physical_cpu_multiplier"] is not None and physical_cpu:
        max_workers = min(
            int(physical_cpu * MULTIPROCESSING_CONFIG["physical_cpu_multiplier"]),
            cpu_count,
        )
    else:
        max_workers = max(
            1, int(cpu_count * MULTIPROCESSING_CONFIG["logical_cpu_percent"])
        )

    # Add resource limitation to prevent memory issues
    # Limit to slightly fewer processes than calculated to leave room for system
    max_workers = max(1, min(max_workers, cpu_count - 1))

    print(f"System has {cpu_count} logical cores and {physical_cpu} physical cores")
    print(f"Using {max_workers} worker processes")

    # Process files in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks, 传递额外的参数到全局函数
        futures = [
            executor.submit(process_single_file_with_lock, csv_file, results, file_lock)
            for csv_file in csv_files
        ]

        # Process results as they complete with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(csv_files),
            desc="Processing files",
        ):
            try:
                # We don't need to do anything here as results are collected in the worker function
                future.result()
            except Exception as e:
                print(f"Error in worker process: {e}")

    # Convert to standard list for further processing
    results = list(results)

    # Create summary DataFrame
    if results:
        results_df = pd.DataFrame(results)

        # Sort by outperformance
        results_df = results_df.sort_values("outperformance", ascending=False)

        # Save results
        results_df.to_csv(f"{results_dir}/backtest_summary.csv", index=False)

        # Print top performers
        print("\n----- Top 10 Performers -----")
        print(
            results_df.head(10)[
                [
                    "symbol",
                    "total_return",
                    "buy_hold_return",
                    "outperformance",
                    "num_trades",
                    "overall_win_rate",
                ]
            ]
        )

        # Print worst performers
        print("\n----- Bottom 10 Performers -----")
        print(
            results_df.tail(10)[
                [
                    "symbol",
                    "total_return",
                    "buy_hold_return",
                    "outperformance",
                    "num_trades",
                    "overall_win_rate",
                ]
            ]
        )

        # 打印多空策略对比
        print("\n----- Long vs Short Strategy Performance -----")
        # 创建多头和空头表现对比的数据框
        long_short_compare = results_df.copy()
        # 只选择有足够多头和空头交易的币种
        long_short_compare = long_short_compare[
            (long_short_compare["long_trades"] >= 5)
            & (long_short_compare["short_trades"] >= 5)
        ]
        # 计算多头和空头的收益率差异
        long_short_compare["long_short_diff"] = (
            long_short_compare["long_avg_profit"]
            - long_short_compare["short_avg_profit"]
        )
        # 按多空收益差排序
        long_short_compare = long_short_compare.sort_values(
            "long_short_diff", ascending=False
        )

        # 打印多头表现更好的币种
        print("\n--- Coins Better for Long Trading ---")
        print(
            long_short_compare.head(10)[
                [
                    "symbol",
                    "long_avg_profit",
                    "long_win_rate",
                    "short_avg_profit",
                    "short_win_rate",
                    "long_short_diff",
                ]
            ]
        )

        # 打印空头表现更好的币种
        print("\n--- Coins Better for Short Trading ---")
        print(
            long_short_compare.tail(10)[
                [
                    "symbol",
                    "long_avg_profit",
                    "long_win_rate",
                    "short_avg_profit",
                    "short_win_rate",
                    "long_short_diff",
                ]
            ]
        )

        # Print average performance
        print("\n----- Average Performance -----")
        print(f"Average Total Return: {results_df['total_return'].mean():.2f}%")
        print(f"Average Buy & Hold Return: {results_df['buy_hold_return'].mean():.2f}%")
        print(f"Average Outperformance: {results_df['outperformance'].mean():.2f}%")
        print(f"Average Number of Trades: {results_df['num_trades'].mean():.2f}")
        print(f"Average Win Rate: {results_df['overall_win_rate'].mean():.2f}%")
        print(f"Average Long Trades: {results_df['long_trades'].mean():.2f}")
        print(f"Average Long Win Rate: {results_df['long_win_rate'].mean():.2f}%")
        print(f"Average Long Profit: {results_df['long_avg_profit'].mean():.2f}%")
        print(f"Average Short Trades: {results_df['short_trades'].mean():.2f}")
        print(f"Average Short Win Rate: {results_df['short_win_rate'].mean():.2f}%")
        print(f"Average Short Profit: {results_df['short_avg_profit'].mean():.2f}%")
        print(f"Average Forced Closes: {results_df['forced_closes'].mean():.2f}")
        print(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}")
        print(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2f}%")

        # 将汇总信息保存到日志文件
        # Use lock for final summary file to ensure it's written atomically
        with file_lock:
            with open(os.path.join(log_dir, "summary_performance.txt"), "w") as f:
                f.write("\n----- Top 10 Performers -----\n")
                f.write(
                    results_df.head(10)[
                        [
                            "symbol",
                            "total_return",
                            "buy_hold_return",
                            "outperformance",
                            "num_trades",
                            "overall_win_rate",
                        ]
                    ].to_string()
                    + "\n\n"
                )

                f.write("\n----- Bottom 10 Performers -----\n")
                f.write(
                    results_df.tail(10)[
                        [
                            "symbol",
                            "total_return",
                            "buy_hold_return",
                            "outperformance",
                            "num_trades",
                            "overall_win_rate",
                        ]
                    ].to_string()
                    + "\n\n"
                )

                # 保存多空策略对比
                if not long_short_compare.empty:
                    f.write("\n----- Long vs Short Strategy Performance -----\n")

                    f.write("\n--- Coins Better for Long Trading ---\n")
                    f.write(
                        long_short_compare.head(10)[
                            [
                                "symbol",
                                "long_avg_profit",
                                "long_win_rate",
                                "short_avg_profit",
                                "short_win_rate",
                                "long_short_diff",
                            ]
                        ].to_string()
                        + "\n\n"
                    )

                    f.write("\n--- Coins Better for Short Trading ---\n")
                    f.write(
                        long_short_compare.tail(10)[
                            [
                                "symbol",
                                "long_avg_profit",
                                "long_win_rate",
                                "short_avg_profit",
                                "short_win_rate",
                                "long_short_diff",
                            ]
                        ].to_string()
                        + "\n\n"
                    )

                f.write("\n----- Average Performance -----\n")
                f.write(
                    f"Average Total Return: {results_df['total_return'].mean():.2f}%\n"
                )
                f.write(
                    f"Average Buy & Hold Return: {results_df['buy_hold_return'].mean():.2f}%\n"
                )
                f.write(
                    f"Average Outperformance: {results_df['outperformance'].mean():.2f}%\n"
                )
                f.write(
                    f"Average Number of Trades: {results_df['num_trades'].mean():.2f}\n"
                )
                f.write(
                    f"Average Win Rate: {results_df['overall_win_rate'].mean():.2f}%\n"
                )
                f.write(
                    f"Average Long Trades: {results_df['long_trades'].mean():.2f}\n"
                )
                f.write(
                    f"Average Long Win Rate: {results_df['long_win_rate'].mean():.2f}%\n"
                )
                f.write(
                    f"Average Long Profit: {results_df['long_avg_profit'].mean():.2f}%\n"
                )
                f.write(
                    f"Average Short Trades: {results_df['short_trades'].mean():.2f}\n"
                )
                f.write(
                    f"Average Short Win Rate: {results_df['short_win_rate'].mean():.2f}%\n"
                )
                f.write(
                    f"Average Short Profit: {results_df['short_avg_profit'].mean():.2f}%\n"
                )
                f.write(
                    f"Average Forced Closes: {results_df['forced_closes'].mean():.2f}\n"
                )
                f.write(
                    f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}\n"
                )
                f.write(
                    f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2f}%\n"
                )

                f.write(
                    f"\n回测结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

                # Add execution time
                execution_time = time.time() - start_time
                f.write(
                    f"\n总执行时间: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)\n"
                )
                f.write(
                    f"平均每个文件处理时间: {execution_time/len(csv_files):.2f} 秒\n"
                )
                f.write(f"总处理文件数: {len(csv_files)}\n")
                f.write(f"有效结果数: {len(results)}\n")

    # Print execution time
    execution_time = time.time() - start_time
    print(
        f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
    )
    print(f"Average time per file: {execution_time/len(csv_files):.2f} seconds")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Valid results: {len(results)}")


if __name__ == "__main__":
    # Ensure this guard for multiprocessing
    main()
