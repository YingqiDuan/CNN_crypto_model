import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculatings import ema, kdj

# 设定数据集路径
data_path = "merged_csv/ETHUSDT_1h_2023-11-23_to_2024-11-20.csv"

# 读取数据
df = pd.read_csv(data_path, parse_dates=True, index_col="open_time")
df = df[["close", "high", "low"]]

# 计算EMA
short_window = 10
long_window = 25
df["EMA_short"] = ema(df, short_window)
df["EMA_long"] = ema(df, long_window)

# 计算KDJ指标
df["K"], df["D"], df["J"] = kdj(df)

# 删除前100行数据
df = df.iloc[100:].copy()

# 识别金叉和死叉
df["Crossover"] = 0
df["Crossover"] = np.where(
    (df["EMA_short"] > df["EMA_long"])
    & (df["EMA_short"].shift(1) <= df["EMA_long"].shift(1)),
    1,
    df["Crossover"],
)
df["Crossover"] = np.where(
    (df["EMA_short"] < df["EMA_long"])
    & (df["EMA_short"].shift(1) >= df["EMA_long"].shift(1)),
    -1,
    df["Crossover"],
)

# 生成交易信号
# 1 表示做多，-1 表示做空，0 表示保持当前仓位
df["Signal"] = df["Crossover"]

# 初始化持仓
df["Position"] = 0
df["Position"] = df["Signal"].replace(
    to_replace=0, method="ffill"
)  # 持仓根据最新信号前移
df["Position"].fillna(0, inplace=True)  # 填充初始仓位为0（无仓位）

# 添加止损逻辑
stop_loss_threshold = 0.01  # 1%的止损阈值

# 初始化开仓价格
df["Entry_Price"] = 0
df["Entry_Price"] = np.where(df["Signal"] != 0, df["close"], np.nan)
df["Entry_Price"] = df["Entry_Price"].ffill()

# 检查止损条件
df["Stop_Loss"] = 0  # 初始化止损信号
df["Stop_Loss"] = np.where(
    (df["Position"] > 0)
    & ((df["close"] < df["Entry_Price"] * (1 - stop_loss_threshold))),
    -1,  # 平多仓
    df["Stop_Loss"],
)
df["Stop_Loss"] = np.where(
    (df["Position"] < 0)
    & ((df["close"] > df["Entry_Price"] * (1 + stop_loss_threshold))),
    1,  # 平空仓
    df["Stop_Loss"],
)

# 更新信号：止损优先于金叉/死叉信号
df["Signal"] = np.where(df["Stop_Loss"] != 0, 0, df["Signal"])  # 止损时关闭信号
df["Signal"] = df["Signal"] + df["Stop_Loss"]

# 更新持仓
df["Position"] = 0
df["Position"] = df["Signal"].replace(to_replace=0, method="ffill")
df["Position"].fillna(0, inplace=True)

# 计算每日收益率
df["Market_Return"] = df["close"].pct_change()
df["Strategy_Return"] = df["Market_Return"] * df["Position"].shift(
    1
)  # 使用前一天的持仓
df["Strategy_Return"].fillna(0, inplace=True)  # 填充NaN值为0

# 计算累计收益率
df["Cumulative_Market_Return"] = (1 + df["Market_Return"]).cumprod() - 1
df["Cumulative_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod() - 1

# 输出最终的累计收益率
total_market_return = df["Cumulative_Market_Return"].iloc[-1]
total_strategy_return = df["Cumulative_Strategy_Return"].iloc[-1]

print(f"市场累计收益率: {total_market_return:.2%}")
print(f"策略累计收益率: {total_strategy_return:.2%}")

# 可视化结果
plt.figure(figsize=(14, 7))
plt.plot(df.index, df["Cumulative_Market_Return"], label="Cumulative_Market_Return")
plt.plot(df.index, df["Cumulative_Strategy_Return"], label="Cumulative_Strategy_Return")
plt.title("EMA10 EMA25 with Stop Loss vs Market")
plt.xlabel("time")
plt.ylabel("Cumulative_Return")
plt.legend()
plt.grid(True)
plt.savefig("EMA10 EMA25 vs Market Cumulative_Return_with_Stop_Loss.png")
plt.show()

# 额外：显示买卖信号
plt.figure(figsize=(14, 7))
plt.plot(df["close"], label="close", alpha=0.5)
plt.plot(df["EMA_short"], label="EMA10", alpha=0.9)
plt.plot(df["EMA_long"], label="EMA25", alpha=0.9)

# 标记做多信号（金叉）
buy_signals = df[df["Crossover"] == 1]
plt.scatter(
    buy_signals.index,
    buy_signals["close"],
    marker="^",
    color="g",
    label="long",
    alpha=1,
)

# 标记做空信号（死叉）
sell_signals = df[df["Crossover"] == -1]
plt.scatter(
    sell_signals.index,
    sell_signals["close"],
    marker="v",
    color="r",
    label="short",
    alpha=1,
)

plt.title("close and EMA10 EMA25 and signal")
plt.xlabel("time")
plt.ylabel("price")
plt.legend()
plt.grid(True)
plt.savefig("close and EMA10 EMA25 and signal_with_Stop_Loss.png")
plt.show()

# 保存结果到CSV文件
output_path = "EMA_Strategy_with_Stop_Loss_Result.csv"
df.to_csv(output_path)
print(f"策略结果（包含止损）已保存到 {output_path}")
