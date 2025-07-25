from binance.um_futures import UMFutures
from indicators import sma, ema, bb, kdj, macd
from predict import get_data, clean_data
from binance.lib.utils import config_logging
from get_trading_pairs import coins


def f1(client, length):
    for coin in coins:
        i = "4h"
        df = get_data(client, coin, i, length)

        if len(df) < length:
            continue

        df = clean_data(df)

        df["ema10"], df["ema25"] = ema(df, 10), ema(df, 25)

        if (
            df["ema10"].iloc[-1] > df["ema25"].iloc[-1]
            and df["ema10"].iloc[-2] < df["ema25"].iloc[-2]
        ):
            print(coin, "up")
        if (
            df["ema10"].iloc[-1] < df["ema25"].iloc[-1]
            and df["ema10"].iloc[-2] > df["ema25"].iloc[-2]
        ):
            print(coin, "down")


def f2(client, length):
    for coin in coins:
        a = 0
        for i in ["4h", "6h", "8h"]:
            df = get_data(client, coin, i, length)

            if len(df) < length:
                break

            df = clean_data(df)

            df["ema10"], df["ema25"] = ema(df, 10), ema(df, 25)

            if df["ema10"].iloc[-1] > df["ema25"].iloc[-1]:
                a += 1
            if df["ema10"].iloc[-1] < df["ema25"].iloc[-1]:
                a -= 1
        if a == 3:
            print(coin, "up")
        elif a == -3:
            print(coin, "down")


def f3(client, length, interval):
    for coin in coins:
        df = get_data(client, coin, interval, length)

        if len(df) < length:
            continue

        df = clean_data(df)

        df["ema7"] = ema(df, 7)
        df["upband"], df["downband"] = bb(df, 20, 0.8)

        if (
            df["ema7"].iloc[-2] > df["upband"].iloc[-2]
            and df["ema7"].iloc[-3] < df["upband"].iloc[-3]
        ):
            print(coin, "up")
            continue

        if (
            df["ema7"].iloc[-2] < df["downband"].iloc[-2]
            and df["ema7"].iloc[-3] > df["downband"].iloc[-3]
        ):
            print(coin, "down")


def f4(client, length, intervals: list[str]):
    for coin in coins:
        a = 0
        for i in intervals:
            df = get_data(client, coin, i, length)

            if len(df) < length:
                continue

            df = clean_data(df)

            df["ema7"] = ema(df, 7)
            df["upband"], df["downband"] = bb(df, 20, 0.8)

            if i == intervals[0]:
                if (
                    df["ema7"].iloc[-1] > df["upband"].iloc[-1]
                    and df["ema7"].iloc[-2] < df["upband"].iloc[-2]
                ):
                    a += 1
                    continue
            elif df["ema7"].iloc[-1] > df["upband"].iloc[-1]:
                a += 1
                continue

            if i == intervals[0]:
                if (
                    df["ema7"].iloc[-1] < df["downband"].iloc[-1]
                    and df["ema7"].iloc[-2] > df["downband"].iloc[-2]
                ):
                    a -= 1
                    continue
            elif df["ema7"].iloc[-1] < df["downband"].iloc[-1]:
                a -= 1
                continue
        if a == 4:
            print(coin, "up")
        if a == -4:
            print(coin, "down")


def f5(client, length, interval):
    for coin in coins:
        df = get_data(client, coin, interval, length)

        if len(df) < length:
            continue

        df = clean_data(df)

        df["ema"] = ema(df, 3)
        df["sma"] = sma(df, 3)
        df["upband"], df["downband"] = bb(df, 20, 0.8)
        df["kdj_k"], df["kdj_d"], df["kdj_j"] = kdj(df, 9, 3, 3)
        df["macd"] = macd(df)

        if (
            df["ema"].iloc[-1] > df["sma"].iloc[-1]
            and df["ema"].iloc[-2] < df["sma"].iloc[-2]
            and df["ema"].iloc[-1] > df["ema"].iloc[-2]
            and df["sma"].iloc[-1] > df["sma"].iloc[-2]
            and df["close"].iloc[-1] - df["open"].iloc[-1] > 0
            and df["upband"].iloc[-1] > df["upband"].iloc[-2]
            and df["downband"].iloc[-1] > df["downband"].iloc[-2]
            and df["ema"].iloc[-2] > df["upband"].iloc[-2]
            and df["kdj_k"].iloc[-1] > df["kdj_d"].iloc[-1]
            and df["macd"].iloc[-1] > 0
        ):
            print(coin, "up")
            continue

        if (
            df["ema"].iloc[-1] < df["sma"].iloc[-1]
            and df["ema"].iloc[-2] > df["sma"].iloc[-2]
            and df["ema"].iloc[-1] < df["ema"].iloc[-2]
            and df["sma"].iloc[-1] < df["sma"].iloc[-2]
            and df["close"].iloc[-1] - df["open"].iloc[-1] < 0
            and df["upband"].iloc[-1] < df["upband"].iloc[-2]
            and df["downband"].iloc[-1] < df["downband"].iloc[-2]
            and df["ema"].iloc[-2] < df["downband"].iloc[-2]
            and df["kdj_k"].iloc[-1] < df["kdj_d"].iloc[-1]
            and df["macd"].iloc[-1] < 0
        ):
            print(coin, "down")


def main():
    client = UMFutures()
    length = 200
    for i in ["4h"]:
        print(i)
        f5(client, length, i)


if __name__ == "__main__":
    main()
