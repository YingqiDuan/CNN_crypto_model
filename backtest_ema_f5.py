import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob


# Import the indicator functions
def sma(df, window):
    return df["close"].rolling(window=window).mean()


def ema(df, window):
    return df["close"].ewm(span=window, adjust=False).mean()


def clean_data(df):
    """Clean and preprocess the data from CSV files"""
    # Convert timestamp to datetime
    df["open_time"] = pd.to_datetime(df["open_time"])

    # Set index to datetime
    df = df.set_index("open_time")

    # Ensure numeric columns are float
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


def f5_strategy(df):
    """
    Implements the f5 strategy from ema.py:
    - Checks for EMA5 crossing above SMA5 (buy signal)
    - Checks for EMA5 crossing below SMA5 (sell signal)
    - Confirms trend direction

    Returns DataFrame with signal column (1 for buy, -1 for sell, 0 for no action)
    """
    # Calculate indicators
    df["ema5"] = ema(df, 5)
    df["sma5"] = sma(df, 5)

    # Initialize signal column
    df["signal"] = 0

    # Buy signals (1)
    buy_condition = (
        (df["ema5"].shift(1) > df["sma5"].shift(1))
        & (df["ema5"].shift(2) < df["sma5"].shift(2))
        & (df["ema5"] > df["sma5"])
        & (df["ema5"] > df["ema5"].shift(1))
        & (df["ema5"].shift(1) > df["ema5"].shift(2))
        & (df["sma5"] > df["sma5"].shift(1))
        & (df["sma5"].shift(1) > df["sma5"].shift(2))
    )

    # Sell signals (-1)
    sell_condition = (
        (df["ema5"].shift(1) < df["sma5"].shift(1))
        & (df["ema5"].shift(2) > df["sma5"].shift(2))
        & (df["ema5"] < df["sma5"])
        & (df["ema5"] < df["ema5"].shift(1))
        & (df["ema5"].shift(1) < df["ema5"].shift(2))
        & (df["sma5"] < df["sma5"].shift(1))
        & (df["sma5"].shift(1) < df["sma5"].shift(2))
    )

    # Set signals
    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1

    return df


def backtest(
    df, initial_capital=10000.0, position_size=0.95, fee_rate=0.0004, max_hold_periods=3
):
    """
    Backtest the strategy on historical data

    Parameters:
    - df: DataFrame with price data and signals
    - initial_capital: Starting capital amount
    - position_size: Percentage of capital to allocate per trade (0-1)
    - fee_rate: Trading fee rate (e.g., 0.0004 = 0.04%)
    - max_hold_periods: Maximum number of periods to hold a position

    Returns:
    - DataFrame with backtesting results
    """
    # Create a copy of the DataFrame to avoid modifying the original
    backtest_df = df.copy()

    # Initialize columns for backtesting
    backtest_df["position"] = 0  # 0 = no position, 1 = long, -1 = short
    backtest_df["entry_price"] = 0.0  # Entry price for current position
    backtest_df["capital"] = initial_capital  # Current capital
    backtest_df["holdings"] = (
        0.0  # Value of current holdings (positive for long, negative for short)
    )
    backtest_df["total_value"] = initial_capital  # Capital + holdings
    backtest_df["hold_periods"] = 0  # Number of periods a position has been held
    backtest_df["forced_close"] = (
        0  # Indicator for forced close due to max hold periods
    )

    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    capital = initial_capital
    holdings = 0
    hold_periods = 0  # 持仓周期计数器

    # Loop through the data to simulate trading
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
            # Calculate position size
            trade_size = capital * position_size
            # Calculate fees
            fee = trade_size * fee_rate
            # Calculate amount after fees
            amount_after_fees = trade_size - fee
            # Calculate shares bought
            shares_bought = amount_after_fees / curr_price
            # Update position
            position = 1
            entry_price = curr_price
            capital -= trade_size
            holdings = shares_bought * curr_price
            hold_periods = 0  # 重置持仓周期计数器

        # 情况2: 当前无仓位，遇到卖出信号，开空仓
        elif signal == -1 and position == 0:
            # Calculate position size
            trade_size = capital * position_size
            # Calculate fees
            fee = trade_size * fee_rate
            # Calculate amount after fees
            amount_after_fees = trade_size - fee
            # Calculate shares shorted
            shares_shorted = amount_after_fees / curr_price
            # Update position
            position = -1
            entry_price = curr_price
            capital -= fee  # 只扣除手续费
            holdings = -shares_shorted * curr_price  # 负值表示空头持仓
            hold_periods = 0  # 重置持仓周期计数器

        # 情况3: 当前持有多仓，遇到卖出信号或强制平仓，平多仓
        elif (signal == -1 or force_close) and position == 1:
            # Calculate value before fees
            value_before_fees = holdings * curr_price / entry_price
            # Calculate fees
            fee = value_before_fees * fee_rate
            # Update capital
            capital += value_before_fees - fee

            # 如果有卖出信号，则直接开空仓
            if signal == -1 and not force_close:
                # Calculate new position size
                trade_size = capital * position_size
                # Calculate fees
                fee = trade_size * fee_rate
                # Calculate amount after fees
                amount_after_fees = trade_size - fee
                # Calculate shares shorted
                shares_shorted = amount_after_fees / curr_price
                # Update position
                position = -1
                entry_price = curr_price
                capital -= fee  # 只扣除手续费
                holdings = -shares_shorted * curr_price  # 负值表示空头持仓
                hold_periods = 0  # 重置持仓周期计数器
            else:
                # 强制平仓，不开新仓位
                position = 0
                entry_price = 0
                holdings = 0
                hold_periods = 0  # 重置持仓周期计数器

        # 情况4: 当前持有空仓，遇到买入信号或强制平仓，平空仓
        elif (signal == 1 or force_close) and position == -1:
            # Calculate profit/loss - for short positions, profit when price decreases
            profit = -holdings * (1 - curr_price / entry_price)
            # Calculate fees
            fee = abs(profit) * fee_rate
            # Update capital
            capital += profit - fee

            # 如果有买入信号，则直接开多仓
            if signal == 1 and not force_close:
                # Calculate new position size
                trade_size = capital * position_size
                # Calculate fees
                fee = trade_size * fee_rate
                # Calculate amount after fees
                amount_after_fees = trade_size - fee
                # Calculate shares bought
                shares_bought = amount_after_fees / curr_price
                # Update position
                position = 1
                entry_price = curr_price
                capital -= trade_size
                holdings = shares_bought * curr_price
                hold_periods = 0  # 重置持仓周期计数器
            else:
                # 强制平仓，不开新仓位
                position = 0
                entry_price = 0
                holdings = 0
                hold_periods = 0  # 重置持仓周期计数器

        # 更新持仓价值
        if position == 1:
            # 多头持仓价值随价格变化
            holdings = holdings * curr_price / backtest_df["close"].iloc[i - 1]
            hold_periods += 1
        elif position == -1:
            # 空头持仓价值随价格变化（反向）
            holdings = holdings * curr_price / backtest_df["close"].iloc[i - 1]
            hold_periods += 1

        # Update backtest DataFrame
        backtest_df.loc[backtest_df.index[i], "position"] = position
        backtest_df.loc[backtest_df.index[i], "entry_price"] = entry_price
        backtest_df.loc[backtest_df.index[i], "capital"] = capital
        backtest_df.loc[backtest_df.index[i], "holdings"] = holdings
        backtest_df.loc[backtest_df.index[i], "total_value"] = capital + abs(
            holdings
        )  # 空头持仓也计入总价值
        backtest_df.loc[backtest_df.index[i], "hold_periods"] = hold_periods

    # Calculate returns and metrics
    backtest_df["return"] = (
        backtest_df["total_value"] / backtest_df["total_value"].iloc[0] - 1
    )
    backtest_df["benchmark_return"] = (
        backtest_df["close"] / backtest_df["close"].iloc[0] - 1
    )

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

    # Calculate annualized return (assuming 252 trading days in a year)
    days = (backtest_df.index[-1] - backtest_df.index[0]).days
    years = days / 365
    annualized_return = (
        ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    )

    # Calculate max drawdown
    backtest_df["cummax"] = backtest_df["total_value"].cummax()
    backtest_df["drawdown"] = (
        backtest_df["total_value"] / backtest_df["cummax"] - 1
    ) * 100
    max_drawdown = backtest_df["drawdown"].min()

    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    risk_free_rate = 0
    if backtest_df["daily_return"].std() != 0:
        sharpe_ratio = (
            (backtest_df["daily_return"].mean() - risk_free_rate)
            / backtest_df["daily_return"].std()
            * (252**0.5)
        )
    else:
        sharpe_ratio = 0

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
    log_dir = "backtest_result_f5/logs"
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
    plt.plot(backtest_df.index, backtest_df["ema5"], label="EMA(5)", color="blue")
    plt.plot(backtest_df.index, backtest_df["sma5"], label="SMA(5)", color="red")

    # 标记仓位变化点
    # 通过计算position列的差分来获取仓位的变化
    backtest_df["position_change"] = backtest_df["position"].diff()

    # 多头开仓 (0->1)
    long_entries = backtest_df[
        (backtest_df["position_change"] == 1) & (backtest_df["position"] == 1)
    ]
    plt.scatter(
        long_entries.index,
        long_entries["close"],
        color="green",
        marker="^",
        alpha=1,
        label="Long Entry",
        s=100,
    )

    # 多头平仓 (1->0 或 1->-1)
    long_exits = backtest_df[
        (backtest_df["position_change"] < 0)
        & (backtest_df["position_change"].shift(1) == 1)
    ]
    plt.scatter(
        long_exits.index,
        long_exits["close"],
        color="red",
        marker="v",
        alpha=1,
        label="Long Exit",
        s=100,
    )

    # 空头开仓 (0->-1)
    short_entries = backtest_df[
        (backtest_df["position_change"] == -1) & (backtest_df["position"] == -1)
    ]
    plt.scatter(
        short_entries.index,
        short_entries["close"],
        color="red",
        marker="v",
        alpha=1,
        label="Short Entry",
        s=100,
    )

    # 空头平仓 (-1->0 或 -1->1)
    short_exits = backtest_df[
        (backtest_df["position_change"] > 0)
        & (backtest_df["position_change"].shift(1) == -1)
    ]
    plt.scatter(
        short_exits.index,
        short_exits["close"],
        color="green",
        marker="^",
        alpha=1,
        label="Short Exit",
        s=100,
    )

    # Plot forced close points
    if "forced_close" in backtest_df.columns:
        forced_close = backtest_df[backtest_df["forced_close"] == 1]
        if not forced_close.empty:
            plt.scatter(
                forced_close.index,
                forced_close["close"],
                color="purple",
                marker="x",
                alpha=1,
                label="Forced Close (3 periods)",
                s=100,
            )

    plt.title(
        f"{symbol} Price with EMA-SMA Strategy (Long & Short, Max Hold: 3 periods)"
    )
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

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

    # 添加仓位区域标记
    for i in range(1, len(backtest_df)):
        if backtest_df["position"].iloc[i] == 1:
            plt.axvspan(
                backtest_df.index[i - 1], backtest_df.index[i], color="green", alpha=0.1
            )
        elif backtest_df["position"].iloc[i] == -1:
            plt.axvspan(
                backtest_df.index[i - 1], backtest_df.index[i], color="red", alpha=0.1
            )

    plt.title(f"{symbol} Strategy Performance vs Buy & Hold")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(True)

    # Plot 3: Drawdowns
    plt.subplot(3, 1, 3)
    plt.fill_between(
        backtest_df.index, backtest_df["drawdown"], 0, color="red", alpha=0.3
    )
    plt.title(f"{symbol} Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"backtest_result_f5/backtest_results_{symbol}.png")
    plt.close()


def main():
    # 创建结果目录
    results_dir = "backtest_result_f5"
    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 清空汇总日志文件
    with open(os.path.join(log_dir, "all_performances.txt"), "w") as f:
        f.write(f"回测开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n\n")

    # Directory with CSV files
    data_dir = "merged_csv"

    # Get list of CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*_15m_*.csv"))

    # Limit to first 5 files for testing (remove this line to process all files)
    # csv_files = csv_files[:5]

    # Initialize results list
    results = []

    # Process each file
    for csv_file in csv_files:
        try:
            symbol = os.path.basename(csv_file).split("_")[0]
            print(f"\nProcessing {symbol}...")

            # Read CSV file
            df = pd.read_csv(csv_file)

            # Clean data
            df = clean_data(df)

            # Skip if not enough data
            if len(df) < 100:
                print(f"Skipping {symbol} - not enough data")
                continue

            # Apply f5 strategy
            df = f5_strategy(df)

            # Run backtest
            backtest_df = backtest(df)

            # Analyze results
            result = analyze_results(backtest_df, symbol)
            results.append(result)

            # Plot results
            plot_results(backtest_df, symbol)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    # Create summary DataFrame
    if results:
        results_df = pd.DataFrame(results)

        # Sort by outperformance
        results_df = results_df.sort_values("outperformance", ascending=False)

        # Save results
        results_df.to_csv("backtest_summary.csv", index=False)

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
            f.write(f"Average Total Return: {results_df['total_return'].mean():.2f}%\n")
            f.write(
                f"Average Buy & Hold Return: {results_df['buy_hold_return'].mean():.2f}%\n"
            )
            f.write(
                f"Average Outperformance: {results_df['outperformance'].mean():.2f}%\n"
            )
            f.write(
                f"Average Number of Trades: {results_df['num_trades'].mean():.2f}\n"
            )
            f.write(f"Average Win Rate: {results_df['overall_win_rate'].mean():.2f}%\n")
            f.write(f"Average Long Trades: {results_df['long_trades'].mean():.2f}\n")
            f.write(
                f"Average Long Win Rate: {results_df['long_win_rate'].mean():.2f}%\n"
            )
            f.write(
                f"Average Long Profit: {results_df['long_avg_profit'].mean():.2f}%\n"
            )
            f.write(f"Average Short Trades: {results_df['short_trades'].mean():.2f}\n")
            f.write(
                f"Average Short Win Rate: {results_df['short_win_rate'].mean():.2f}%\n"
            )
            f.write(
                f"Average Short Profit: {results_df['short_avg_profit'].mean():.2f}%\n"
            )
            f.write(
                f"Average Forced Closes: {results_df['forced_closes'].mean():.2f}\n"
            )
            f.write(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}\n")
            f.write(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2f}%\n")

            f.write(f"\n回测结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
