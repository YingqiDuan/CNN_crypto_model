import uuid
import shutil
import requests
import zipfile
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
from tenacity import retry, wait_fixed, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle
import numpy as np


class DataProcessor:
    def __init__(self):
        self.session = requests.Session()
        self.ALLOWED_PERIODS = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]

    @retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
    def download_file(self, url: str, local_path: Path) -> str:
        """
        下载文件并保存到本地。

        :param url: 文件的URL地址
        :param local_path: 保存的本地文件路径
        :return: 下载状态 ('exists', 'downloaded', 'failed')
        """
        if local_path.exists():
            # 如果文件大小大于0，认为已存在
            if local_path.stat().st_size > 0:
                return "exists"
            else:
                # 如果文件为空，删除并重新下载
                local_path.unlink()
        try:
            with self.session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            # 下载完成后再次检查文件大小
            if local_path.stat().st_size == 0:
                print(f"下载的文件 {local_path} 为空，可能下载失败。")
                local_path.unlink()
                return "failed"
            return "downloaded"
        except requests.RequestException:
            if local_path.exists():
                local_path.unlink()
        return "failed"

    def unzip_file(
        self, zip_path: Path, extract_dir: Path, trading_pair: str, period: str
    ) -> str:
        """
        解压ZIP文件，将CSV文件提取到指定目录，并重命名以避免冲突。

        :param zip_path: ZIP文件路径
        :param extract_dir: 解压目标目录
        :param trading_pair: 交易对名称
        :param period: 周期
        :return: 解压状态 ('unzipped', 'bad_zip', 'failed')
        """
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                if not zip_ref.namelist():
                    print(f"ZIP 文件 {zip_path} 中没有任何文件。")
                    zip_path.unlink()
                    return "bad_zip"
                date_str = zip_path.stem.split("-")[-1]
                for file in zip_ref.namelist():
                    if file.endswith(".csv"):
                        unique_id = uuid.uuid4().hex[:8]
                        new_filename = f"{trading_pair}_{period}_{date_str}_{unique_id}_{Path(file).name}"
                        dst_path = extract_dir / new_filename
                        with zip_ref.open(file) as src, open(dst_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        if dst_path.stat().st_size == 0:
                            print(f"解压得到的文件 {dst_path} 为空，已删除。")
                            dst_path.unlink()
            zip_path.unlink()
            return "unzipped"
        except zipfile.BadZipFile:
            print(f"ZIP 文件 {zip_path} 已损坏，无法解压。")
            zip_path.unlink()
            return "bad_zip"
        except Exception:
            print(f"解压 ZIP 文件 {zip_path} 时出错: {e}")
            return "failed"

    def generate_url(self, trading_pair: str, period: str, date_str: str) -> tuple:
        """
        根据交易对、周期和日期生成下载URL和文件名。

        :param trading_pair: 交易对名称
        :param period: 周期
        :param date_str: 日期字符串
        :return: (URL, 文件名)
        """
        base_url = f"https://data.binance.vision/data/futures/um/daily/klines/{trading_pair}/{period}/"
        filename = f"{trading_pair}-{period}-{date_str}.zip"
        return base_url + filename, filename

    def get_user_input(self) -> tuple:
        """
        获取用户输入的下载天数、交易对列表和周期列表。

        :return: (天数, 交易对列表, 周期列表)
        """
        while True:
            try:
                days = int(input("请输入下载天数（如7）："))
                if days > 0:
                    break
            except ValueError:
                pass
            print("请输入有效天数。")

        while True:
            trading_pairs = input("输入交易对（逗号分隔）：").strip().upper().split(",")
            trading_pairs = [pair.strip() for pair in trading_pairs if pair.strip()]
            if trading_pairs:
                break
            print("请输入至少一个交易对。")

        while True:
            periods = input("输入周期（逗号分隔）：").strip().lower().split(",")
            periods = [
                p.strip() for p in periods if p.strip() and p in self.ALLOWED_PERIODS
            ]
            if periods:
                break
            print("输入无效周期。")

        return days, trading_pairs, periods

    def process_task(self, task: tuple) -> tuple:
        """
        处理单个下载和解压任务。

        :param task: (交易对, 周期, URL, 本地路径, 解压目录)
        :return: (下载状态, 解压状态)
        """
        trading_pair, period, url, local_path, extract_dir = task
        download_status = self.download_file(url, local_path)
        unzip_status = "not_unzipped"
        if download_status in ("downloaded", "exists"):
            unzip_status = self.unzip_file(
                local_path, extract_dir, trading_pair, period
            )
        return download_status, unzip_status

    def download_data(self, days, trading_pairs, periods):
        today = datetime.today()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

        download_base = Path("downloads")
        extract_base = Path("extracted_csv_files")

        # 预创建所有需要的目录，避免在任务处理中重复创建
        for trading_pair in trading_pairs:
            for period in periods:
                (download_base / trading_pair / period).mkdir(
                    parents=True, exist_ok=True
                )
                (extract_base / trading_pair / period).mkdir(
                    parents=True, exist_ok=True
                )

        tasks = []
        for trading_pair in trading_pairs:
            for period in periods:
                for date_str in dates:
                    url, filename = self.generate_url(trading_pair, period, date_str)
                    local_path = download_base / trading_pair / period / filename
                    extract_dir = extract_base / trading_pair / period
                    tasks.append((trading_pair, period, url, local_path, extract_dir))

        failed_downloads = []
        failed_unzips = []

        max_workers = min(10, len(tasks))  # 根据需要调整
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_task, task): task for task in tasks}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="总体进度"
            ):
                task = futures[future]
                try:
                    download_status, unzip_status = future.result()
                    if download_status == "failed":
                        failed_downloads.append(task)
                    if unzip_status in ("bad_zip", "failed"):
                        failed_unzips.append(task)
                except Exception:
                    failed_downloads.append(task)

        shutil.rmtree(download_base, ignore_errors=True)
        print("所有任务完成。")

        if failed_downloads:
            print(f"以下文件下载失败 ({len(failed_downloads)}):")
            for task in failed_downloads:
                print(task[2])  # URL
        if failed_unzips:
            print(f"以下文件解压失败 ({len(failed_unzips)}):")
            for task in failed_unzips:
                print(task[3])  # 本地路径

    def merge_csv_files(
        self,
        trading_pair,
        period,
        base_folder="extracted_csv_files",
        output_folder="merged_csv",
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
            return None

        print(f"找到 {len(csv_files)} 个 CSV 文件，开始合并...")

        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if df.empty:
                    print(f"文件 {file} 是空的，已跳过。")
                    continue
                # 删除空行
                df.dropna(how="all", inplace=True)
                dataframes.append(df)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

        if not dataframes:
            print("没有有效的 CSV 文件可供合并。")
            return None

        try:
            combined_df = pd.concat(
                (pd.read_csv(file) for file in csv_files),
                ignore_index=True,
            )
        except Exception as e:
            print(f"合并 CSV 文件时出错: {e}")
            return None

        # 删除合并后的 DataFrame 中的空行
        combined_df.dropna(how="all", inplace=True)

        # 根据 Binance 数据格式，调整列名
        required_columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]

        combined_df = combined_df.iloc[:, :12]
        combined_df.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        combined_df = combined_df.filter(items=required_columns, axis=1)

        try:
            combined_df["open_time"] = pd.to_datetime(
                combined_df["open_time"], unit="ms"
            )
        except Exception as e:
            print(f"转换 'open_time' 为 datetime 类型时出错: {e}")
            return None

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
        return output_file

    def clean_extracted_files(self, folder_path: str = "extracted_csv_files"):
        """
        删除指定的文件夹及其内容。

        :param folder_path: 要删除的文件夹路径，默认为 "extracted_csv_files"
        """
        folder = Path(folder_path)
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
            print(f"已删除 '{folder_path}' 文件夹。")
        else:
            print(f"'{folder_path}' 文件夹不存在或不是一个目录。")

    def merge_data(self, trading_pairs, periods):
        if isinstance(trading_pairs, str):
            trading_pairs = [
                pair.strip().upper()
                for pair in trading_pairs.split(",")
                if pair.strip()
            ]

        if not trading_pairs:
            print("未输入有效的交易对。程序将退出。")
            return []
        if isinstance(periods, str):
            periods = [
                period.strip() for period in periods.split(",") if period.strip()
            ]

        if not periods:
            print("未输入有效的周期。程序将退出。")
            return []

        base_folder = "extracted_csv_files"
        output_folder = "merged_csv"
        merged_files = []

        for trading_pair in trading_pairs:
            for period in periods:
                print(f"\n正在处理交易对: {trading_pair}, 周期: {period}")
                output_file = self.merge_csv_files(
                    trading_pair, period, base_folder, output_folder
                )
                if output_file:
                    merged_files.append(output_file)

        # 调用类方法删除 extracted_csv_files 文件夹
        self.clean_extracted_files()
        return merged_files

    def split_data(self, csv_path: Path, save_path: Path):
        """
        将 CSV 文件拆分为多个 100 行的样本，并为每个样本生成对应的标签。
        同时，对每个样本的指定列进行归一化处理。

        :param csv_path: 输入的 CSV 文件路径
        :param save_path: 输出的 pickle 文件路径
        """
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_path)
            # 删除空行
            df.dropna(how="all", inplace=True)

            df.reset_index(drop=True, inplace=True)

            # 检查是否还有数据
            if df.empty:
                print(f"CSV 文件 {csv_path} 为空或只有空行，跳过处理。")
                return

            # 初始化存储样本和标签的列表
            samples = []
            labels = []

            # 定义需要归一化的列
            normalize_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
            ]

            # 检查所有需要归一化的列是否存在
            missing_columns = [
                col for col in normalize_columns if col not in df.columns
            ]
            if missing_columns:
                print(f"CSV 文件中缺少以下必要列: {missing_columns}")
                return

            # 提取不同条件的样本
            for idx in range(100, len(df)):
                # 提取当前索引的前 100 行
                matrix = df.iloc[idx - 100 : idx].reset_index(drop=True)
                # 删除时间列
                matrix = matrix.drop(columns=["open_time"])

                # 如果样本中存在任何空值，跳过该样本
                if matrix.isnull().values.any():
                    continue

                # 获取每个需要归一化列的第一个值
                first_values = {}
                skip_sample = False
                for col in normalize_columns:
                    first_val = matrix.loc[0, col]
                    if first_val == 0:
                        print(f"第 {idx} 行的列 '{col}' 的第一个值为 0，跳过该样本。")
                        skip_sample = True
                        break
                    first_values[col] = first_val

                if skip_sample:
                    continue

                # 对指定列进行归一化
                for col in normalize_columns:
                    matrix[col] = matrix[col] / first_values[col]

                # 根据条件设置 label
                current_close = df.loc[idx, "close"]
                current_open = df.loc[idx, "open"]
                if current_close > current_open * 1.01:
                    label = 1
                elif current_close < current_open * 0.99:
                    label = -1
                else:
                    label = 0

                # 保存样本和对应的 label
                samples.append(matrix)
                labels.append(label)

            if not samples:
                return

            # 保存样本和标签到文件
            with open(save_path, "wb") as f:
                pickle.dump({"samples": samples, "labels": labels}, f)

            print(f"样本和标签已保存到 {save_path}")

        except Exception as e:
            print(f"处理 CSV 文件时出错: {e}")


if __name__ == "__main__":
    dp = DataProcessor()
    days, trading_pairs, periods = dp.get_user_input()
    dp.download_data(days, trading_pairs, periods)
    merged_files = dp.merge_data(trading_pairs, periods)
    for csv_path in merged_files:
        csv_path = Path(csv_path)
        # 获取原文件名的 stem（不包含后缀）
        stem = csv_path.stem
        # 创建新的文件名，添加 '_samples_with_labels.pkl'
        new_filename = stem + "_samples_with_labels.pkl"
        # 使用 with_name 方法替换文件名
        save_path = csv_path.with_name(new_filename)
        dp.split_data(csv_path, save_path)
