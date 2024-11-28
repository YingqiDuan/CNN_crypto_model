from cnn import (
    load_multiple_pkls,
    standardize_samples,
    load_model,
    predict,
)
import torch


def main():

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件路径
    file_path = "merged_csv/BTCUSDT_1h_2023-11-29_to_2024-11-26_samples_with_labels.pkl"  #  .pkl 文件路径

    # 加载和处理数据
    samples, labels = load_multiple_pkls(file_path)
    if samples is None or labels is None:
        return

    # 标准化样本
    samples = standardize_samples(samples)
    # 加载模型
    model_path = "simple_cnn_model.pth"
    model = load_model(model_path, device)

    # 选择一个样本进行预测
    sample_index = 388  # 选择第一个样本
    sample = samples[sample_index]
    true_label = labels[sample_index]
    predicted_label = predict(model, sample, device)

    print(f"真实标签: {true_label}, 预测标签: {predicted_label}")


if __name__ == "__main__":
    main()
