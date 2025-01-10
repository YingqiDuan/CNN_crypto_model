from binance.um_futures import UMFutures
from indicators import ema
from get_trading_pairs import coins
from predict import get_data, clean_data


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


if __name__ == "__main__":
    client = UMFutures()
    length = 100
    f1(client, length)
