from binance.um_futures import UMFutures
from indicators import ema, bb
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


def main():
    client = UMFutures()
    length = 50
    for i in ["1w", "3d", "1d", "12h", "8h", "6h", "4h"]:
        print(i)
        f3(client, length, i)


if __name__ == "__main__":
    main()
