import pandas as pd
import numpy as np


def sma(df, window):
    return df["close"].rolling(window=window).mean()


def ema(df, window):
    return df["close"].ewm(span=window, adjust=False).mean()


def vol_sma(df, window):
    return df["volume"].rolling(window=window).mean()


def macd(df, short_window=12, long_window=26, signal_window=9):
    ema_short = ema(df, short_window)
    ema_long = ema(df, long_window)
    dif = ema_short - ema_long
    dea = dif.ewm(span=signal_window, adjust=False).mean()
    return dif - dea, dif, dea


def bb(df, window=20, std_num=1.949):
    mid = sma(df, window)
    std = df["close"].rolling(window=window).std()
    upper = mid + std * std_num
    lower = mid - std * std_num
    return upper, lower


def kdj(df, window=9, k_s=3, d_s=3):
    high = df["high"].rolling(window=window, min_periods=1).max()
    low = df["low"].rolling(window=window, min_periods=1).min()
    rsv = (df["close"] - low) / (high - low) * 100
    k = rsv.ewm(alpha=1 / k_s, adjust=False).mean()
    d = k.ewm(alpha=1 / d_s, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


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
