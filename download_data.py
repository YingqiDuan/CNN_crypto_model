import os
import uuid
import shutil
import requests
import zipfile
from tqdm import tqdm
from datetime import datetime, timedelta
from tenacity import retry, wait_fixed, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

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
def download_file(url, local_filename):
    if os.path.exists(local_filename):
        print(f"{local_filename} 已存在，跳过下载。")
        return local_filename
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        return local_filename
    except requests.RequestException as e:
        print(f"下载失败: {e}")
        return None


def unzip_file(zip_path, extract_to, trading_pair, period):
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            date_str = zip_path.split("-")[-1].split(".")[0]
            for file in zip_ref.namelist():
                if file.endswith(".csv"):
                    new_filename = f"{trading_pair}_{period}_{date_str}_{uuid.uuid4().hex[:8]}_{os.path.basename(file)}"
                    with zip_ref.open(file) as src, open(
                        os.path.join(extract_to, new_filename), "wb"
                    ) as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(zip_path)
    except zipfile.BadZipFile:
        print(f"无效的ZIP文件: {zip_path}")


def generate_url(trading_pair, period, date_str):
    base_url = f"https://data.binance.vision/data/futures/um/daily/klines/{trading_pair}/{period}/"
    filename = f"{trading_pair}-{period}-{date_str}.zip"
    return base_url + filename, filename


def get_user_input():
    while True:
        try:
            days = int(input("请输入下载天数（如7）："))
            if days > 0:
                break
        except ValueError:
            pass
        print("请输入有效天数。")
    trading_pairs = input("输入交易对（逗号分隔）：").strip().upper().split(",")
    periods = input("输入周期（逗号分隔）：").strip().lower().split(",")
    periods = [p for p in periods if p in ALLOWED_PERIODS]
    if not periods:
        print("输入无效周期。")
        return get_user_input()
    return days, trading_pairs, periods


def main():
    days, trading_pairs, periods = get_user_input()
    dates = [
        (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)
    ]
    tasks = []

    for trading_pair in trading_pairs:
        for period in periods:
            for date_str in dates:
                url, filename = generate_url(trading_pair, period, date_str)
                local_path = os.path.join("downloads", trading_pair, period, filename)
                extract_dir = os.path.join("extracted_csv_files", trading_pair, period)
                tasks.append(
                    (
                        trading_pair,
                        period,
                        url,
                        local_path,
                        extract_dir,
                    )
                )

    with ThreadPoolExecutor(max_workers=min(10, len(tasks))) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="总体进度"):
            pass

    shutil.rmtree("downloads", ignore_errors=True)
    print("所有任务完成。")


def process_task(task):
    trading_pair, period, url, local_path, extract_dir = task
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    downloaded_file = download_file(url, local_path)
    if downloaded_file:
        unzip_file(downloaded_file, extract_dir, trading_pair, period)


if __name__ == "__main__":
    main()
