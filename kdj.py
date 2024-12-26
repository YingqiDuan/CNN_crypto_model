from binance.um_futures import UMFutures
from indicators import kdj
from get_trading_pairs import coins
from predict import get_data, clean_data


if __name__ == "__main__":
    client = UMFutures()

    length = 30

    for coin in coins:
        e = 0
        for i in ["15m", "30m", "1h", "2h", "4h"]:
            df = get_data(client, coin, i, length)

            if len(df) < length:
                continue

            df = clean_data(df)

            df["k"], df["d"], df["j"] = kdj(df)

            if df["d"].iloc[-1] > df["k"].iloc[-1] > df["j"].iloc[-1]:
                e += 1
            if df["j"].iloc[-1] > df["k"].iloc[-1] > df["d"].iloc[-1]:
                e -= 1
        if e == 3:
            print(coin, 1)
        if e == -3:
            print(coin, -1)
