def sma(df, window):
    return df["close"].rolling(window=window).mean()


def ema(df, window):
    return df["close"].ewm(span=window, adjust=False).mean()


def macd(df, short_window=12, long_window=26, signal_window=9):
    ema_short = ema(df, short_window)
    ema_long = ema(df, long_window)
    dif = ema_short - ema_long
    dea = dif.ewm(span=signal_window, adjust=False).mean()
    return dif - dea


def bb(df, window=20, std_num=1.949):
    mid = sma(df, window)
    std = df["close"].rolling(window=window).std()
    upper = mid + std * std_num
    lower = mid - std * std_num
    return upper, lower


import pandas as pd
import numpy as np


def kdj(df, window=9, k_s=3, d_s=3):
    high = df["high"].rolling(window=window, min_periods=1).max()
    low = df["low"].rolling(window=window, min_periods=1).min()
    rsv = (df["close"] - low) / (high - low) * 100

    k = pd.Series(index=df.index, dtype=float)
    d = pd.Series(index=df.index, dtype=float)
    j = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i < window - 2:
            k.iloc[i] = np.nan
            d.iloc[i] = np.nan
            j.iloc[i] = np.nan
        elif i == window - 2:
            k.iloc[i] = 50
            d.iloc[i] = 50
            j.iloc[i] = 3 * 50 - 2 * 50  # j = 50
        else:
            k.iloc[i] = (1 / k_s) * rsv.iloc[i] + (1 - 1 / k_s) * k.iloc[i - 1]
            d.iloc[i] = (1 / d_s) * k.iloc[i] + (1 - 1 / d_s) * d.iloc[i - 1]
            j.iloc[i] = 3 * k.iloc[i] - 2 * d.iloc[i]
    return k, d, j


def rsi(df, window=14):
    delta = df["close"].diff()
    up = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    down = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + up / down))
