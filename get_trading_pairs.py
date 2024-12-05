from binance.um_futures import UMFutures


def custom_sort(item):
    return tuple(item[i] for i in range(len(item)))


um_futures_client = UMFutures()
a = um_futures_client.exchange_info()
coins = []
for i in a["symbols"]:
    if i["status"] == "TRADING" and i["contractType"] == "PERPETUAL":
        coins.append(i["symbol"])


usdt_coin = [s for s in coins if "USDT" in s]
sorted_list = sorted(usdt_coin, key=custom_sort)
coins = sorted_list
# print(sorted_list)
# print(",".join(sorted_list))
