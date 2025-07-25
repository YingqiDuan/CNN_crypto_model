from data_processor import DataProcessor
import glob
from pathlib import Path


def main():
    dp = DataProcessor()
    data_directory = "merged_csv"
    interval = input("Enter interval: ")

    csv_file = glob.glob(f"{data_directory}/*{interval}*.csv")

    for csv_path in csv_file:
        csv_path = Path(csv_path)
        # 获取原文件名的 stem（不包含后缀）
        stem = csv_path.stem
        # 创建新的文件名，添加 '_samples_with_labels.pkl'
        diff = 0.05
        new_filename = stem + f"_samples_with_labels_{diff}.pkl"
        # 使用 with_name 方法替换文件名
        save_path = csv_path.with_name(new_filename)
        dp.split_data(csv_path, save_path, diff=diff)


if __name__ == "__main__":
    main()
