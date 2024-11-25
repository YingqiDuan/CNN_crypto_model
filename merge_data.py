import pandas as pd
from pathlib import Path
import glob


def merge_csv_files(
    trading_pair, period, base_folder="extracted_csv_files", output_folder="merged_csv"
):
    """
    合并指定交易对和周期的 CSV 文件，并按时间排序，输出文件名包含时间范围。

    参数:
    - trading_pair (str): 交易对，如 'BTCUSDT'
    - period (str): 周期，如 '1m'
    - base_folder (str): CSV 文件的基准目录
    - output_folder (str): 合并后文件的输出目录
    """
    csv_folder = Path(base_folder) / trading_pair / period
    csv_files = list(csv_folder.glob(f"{trading_pair}_{period}_*.csv"))

    if not csv_files:
        print(
            f"在目录 {csv_folder} 中未找到任何 CSV 文件。请检查交易对和周期是否正确。"
        )
        return

    print(f"找到 {len(csv_files)} 个 CSV 文件，开始合并...")

    try:
        combined_df = pd.concat(
            (pd.read_csv(file).assign(source_file=file.name) for file in csv_files),
            ignore_index=True,
        )
    except Exception as e:
        print(f"合并 CSV 文件时出错: {e}")
        return

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]

    combined_df = combined_df.filter(items=required_columns, axis=1)

    try:
        combined_df["open_time"] = pd.to_datetime(combined_df["open_time"], unit="ms")
    except Exception as e:
        print(f"转换 'open_time' 为 datetime 类型时出错: {e}")
        return

    combined_df.sort_values(by="open_time", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    start_time = combined_df["open_time"].min().strftime("%Y-%m-%d")
    end_time = combined_df["open_time"].max().strftime("%Y-%m-%d")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_path / f"{trading_pair}_{period}_{start_time}_to_{end_time}.csv"
    )

    combined_df.to_csv(output_file, index=False)
    print(f"所有文件已成功合并并保存到 {output_file}")


def main():
    trading_pairs = input(
        "请输入交易对（多个交易对请用逗号分隔，例如 BTCUSDT,ETHUSDT）："
    )
    trading_pairs = [
        pair.strip().upper() for pair in trading_pairs.split(",") if pair.strip()
    ]

    if not trading_pairs:
        print("未输入有效的交易对。程序将退出。")
        return

    periods = input("请输入周期（多个周期请用逗号分隔，例如 1m,5m）：")
    periods = [period.strip() for period in periods.split(",") if period.strip()]

    if not periods:
        print("未输入有效的周期。程序将退出。")
        return

    base_folder = "extracted_csv_files"
    output_folder = "merged_csv"

    for trading_pair in trading_pairs:
        for period in periods:
            print(f"\n正在处理交易对: {trading_pair}, 周期: {period}")
            merge_csv_files(trading_pair, period, base_folder, output_folder)


if __name__ == "__main__":
    main()
