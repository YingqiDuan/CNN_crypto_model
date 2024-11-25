import uuid
import shutil
import requests
import zipfile
from tqdm import tqdm
from datetime import datetime, timedelta
from tenacity import retry, wait_fixed, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ALLOWED_PERIODS = [
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
def download_file(url: str, local_path: Path) -> str:
    """
    下载文件并保存到本地。

    :param url: 文件的URL地址
    :param local_path: 保存的本地文件路径
    :return: 下载状态 ('exists', 'downloaded', 'failed')
    """
    if local_path.exists():
        return "exists"
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return "downloaded"
    except requests.RequestException:
        return "failed"


def unzip_file(
    zip_path: Path, extract_dir: Path, trading_pair: str, period: str
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
            date_str = zip_path.stem.split("-")[-1]
            for file in zip_ref.namelist():
                if file.endswith(".csv"):
                    unique_id = uuid.uuid4().hex[:8]
                    new_filename = f"{trading_pair}_{period}_{date_str}_{unique_id}_{Path(file).name}"
                    dst_path = extract_dir / new_filename
                    with zip_ref.open(file) as src, open(dst_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        zip_path.unlink()
        return "unzipped"
    except zipfile.BadZipFile:
        return "bad_zip"
    except Exception:
        return "failed"


def generate_url(trading_pair: str, period: str, date_str: str) -> tuple:
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


def get_user_input() -> tuple:
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
        periods = [p.strip() for p in periods if p.strip() and p in ALLOWED_PERIODS]
        if periods:
            break
        print("输入无效周期。")

    return days, trading_pairs, periods


def process_task(task: tuple) -> tuple:
    """
    处理单个下载和解压任务。

    :param task: (交易对, 周期, URL, 本地路径, 解压目录)
    :return: (下载状态, 解压状态)
    """
    trading_pair, period, url, local_path, extract_dir = task
    download_status = download_file(url, local_path)
    unzip_status = "not_unzipped"
    if download_status in ("downloaded", "exists"):
        unzip_status = unzip_file(local_path, extract_dir, trading_pair, period)
    return download_status, unzip_status


def main():
    days, trading_pairs, periods = get_user_input()
    dates = [
        (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)
    ]

    download_base = Path("downloads")
    extract_base = Path("extracted_csv_files")

    # 预创建所有需要的目录，避免在任务处理中重复创建
    for trading_pair in trading_pairs:
        for period in periods:
            (download_base / trading_pair / period).mkdir(parents=True, exist_ok=True)
            (extract_base / trading_pair / period).mkdir(parents=True, exist_ok=True)

    tasks = []
    for trading_pair in trading_pairs:
        for period in periods:
            for date_str in dates:
                url, filename = generate_url(trading_pair, period, date_str)
                local_path = download_base / trading_pair / period / filename
                extract_dir = extract_base / trading_pair / period
                tasks.append((trading_pair, period, url, local_path, extract_dir))

    failed_downloads = []
    failed_unzips = []

    with ThreadPoolExecutor(max_workers=min(10, len(tasks))) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="总体进度"):
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


if __name__ == "__main__":
    main()
