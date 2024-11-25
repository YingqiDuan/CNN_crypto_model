import pandas as pd
import glob
import os


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
    # 构建目标文件夹路径
    csv_folder = os.path.join(base_folder, trading_pair, period)

    # 使用 glob 获取所有匹配的 CSV 文件路径
    csv_pattern = os.path.join(csv_folder, f"{trading_pair}_{period}_*.csv")
    csv_files = glob.glob(csv_pattern)

    # 检查是否找到任何 CSV 文件
    if not csv_files:
        print(
            f"在目录 {csv_folder} 中未找到任何 CSV 文件。请检查交易对和周期是否正确。"
        )
        return

    print(f"找到 {len(csv_files)} 个 CSV 文件，将开始合并...")

    # 初始化一个空的 DataFrame 列表
    df_list = []

    # 遍历所有 CSV 文件并读取内容
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = os.path.basename(file)  # 可选：添加源文件名列
            df_list.append(df)
            print(f"已读取文件: {file}")
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")

    # 检查是否成功读取任何数据
    if not df_list:
        print("未成功读取任何 CSV 文件。请检查文件内容和格式。")
        return

    # 合并所有 DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

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
    combined_df = combined_df.filter(items=required_columns)

    # 选择时间列，这里使用 'open_time'
    time_column = "open_time"

    if time_column not in combined_df.columns:
        print(f"列 '{time_column}' 不存在于合并后的 DataFrame 中。")
        return

    # 将 'open_time' 转换为 datetime
    try:
        combined_df["open_time"] = pd.to_datetime(combined_df["open_time"], unit="ms")
    except Exception as e:
        print(f"转换 'open_time' 为 datetime 类型时出错: {e}")
        return

    # 确定时间范围
    start_time = combined_df["open_time"].min().strftime("%Y-%m-%d")
    end_time = combined_df["open_time"].max().strftime("%Y-%m-%d")

    # 按 'open_time' 列排序
    combined_df.sort_values(by="open_time", inplace=True)

    # 重置索引
    combined_df.reset_index(drop=True, inplace=True)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构建输出文件路径，包含时间范围
    output_file = os.path.join(
        output_folder, f"{trading_pair}_{period}_{start_time}_to_{end_time}.csv"
    )

    # 导出合并后的 CSV 文件
    combined_df.to_csv(output_file, index=False)

    print(f"所有文件已成功合并并保存到 {output_file}")


def main():
    """
    主函数，处理用户输入的交易对和周期。
    """
    # 获取用户输入的交易对
    trading_pairs_input = input(
        "请输入交易对（多个交易对请用逗号分隔，例如 BTCUSDT,ETHUSDT）："
    )
    trading_pairs = [
        pair.strip().upper() for pair in trading_pairs_input.split(",") if pair.strip()
    ]

    if not trading_pairs:
        print("未输入有效的交易对。程序将退出。")
        return

    # 获取用户输入的周期
    periods_input = input("请输入周期（多个周期请用逗号分隔，例如 1m,5m）：")
    periods = [period.strip() for period in periods_input.split(",") if period.strip()]

    if not periods:
        print("未输入有效的周期。程序将退出。")
        return

    # 基准目录和输出目录预定义
    base_folder = "extracted_csv_files"  # 基准目录
    output_folder = "merged_csv"  # 输出目录

    # 处理每个交易对和周期的组合
    for trading_pair in trading_pairs:
        for period in periods:
            print(f"\n正在处理交易对: {trading_pair}, 周期: {period}")
            merge_csv_files(trading_pair, period, base_folder, output_folder)


if __name__ == "__main__":
    main()
