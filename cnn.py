import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 1. 数据加载与预处理
def load_and_process_pkl(file_path):
    """
    读取指定的 .pkl 文件，提取 'samples' 和 'labels'，去除每个样本矩阵的第一列，并返回处理后的数据。

    参数：
        file_path (str): .pkl 文件的路径。

    返回:
        tuple: 包含处理后的 samples 和 labels 的元组。
            - samples (numpy.ndarray): 形状为 (样本数量, 100, 9) 的数组。
            - labels (numpy.ndarray): 形状为 (样本数量,) 的数组。
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        if not isinstance(data, dict):
            raise TypeError("加载的数据不是字典类型。")

        if "samples" not in data or "labels" not in data:
            raise KeyError("字典中缺少 'samples' 或 'labels' 键。")

        samples = data["samples"]
        labels = data["labels"]

        samples = np.array(samples)
        labels = np.array(labels)

        if samples.ndim != 3 or samples.shape[1] != 100 or samples.shape[2] != 9:
            raise ValueError("每个样本的矩阵形状应为 (100, 9)。")

        return samples, labels

    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
    except pickle.UnpicklingError:
        print("无法解序列化文件，文件可能已损坏或格式不正确。")
    except KeyError as e:
        print(f"字典中缺少必要的键: {e}")
    except TypeError as e:
        print(f"数据类型错误: {e}")
    except ValueError as e:
        print(f"值错误: {e}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")


def map_labels(labels):
    """
    将标签从 {-1, 0, 1} 映射到 {0, 1, 2}。

    参数：
        labels (numpy.ndarray): 原始标签数组。

    返回:
        numpy.ndarray: 映射后的标签数组。
    """
    label_mapping = {-1: 0, 0: 1, 1: 2}
    mapped_labels = np.vectorize(label_mapping.get)(labels)
    return mapped_labels


def standardize_samples(samples):
    """
    对 samples 进行标准化处理。

    参数：
        samples (numpy.ndarray): 原始 samples，形状为 (样本数量, 100, 9)。

    返回:
        numpy.ndarray: 标准化后的 samples，形状不变。
    """
    num_samples, rows, cols = samples.shape
    samples_reshaped = samples.reshape(num_samples, -1)  # 转换为 (样本数量, 900)

    scaler = StandardScaler()
    samples_scaled = scaler.fit_transform(samples_reshaped)

    # 恢复原始形状
    samples_scaled = samples_scaled.reshape(num_samples, rows, cols)

    return samples_scaled


# 2. 创建自定义数据集
class CustomMatrixDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # 添加 channel 维度
        sample = sample.unsqueeze(0)

        return sample, label


def split_train_test(samples, labels, test_size=0.2, random_state=42):
    """
    将数据集拆分为训练集和测试集。

    参数：
        samples (numpy.ndarray): 样本数据。
        labels (numpy.ndarray): 标签数据。
        test_size (float): 测试集所占比例。
        random_state (int): 随机种子。

    返回:
        tuple: 拆分后的训练集和测试集。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return X_train, X_test, y_train, y_test


# 3. 定义卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1：输入通道=1，输出通道=16，卷积核大小=3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        # 卷积层2：输入通道=16，输出通道=32，卷积核大小=3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        # 丢弃层
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层
        self.fc1 = nn.Linear(32 * 23 * 5, 128)  # 根据新的输出尺寸调整
        self.fc2 = nn.Linear(128, 3)  # 3 个类别

    def forward(self, x):
        # 卷积层1 -> ReLU
        x = F.relu(self.conv1(x))  # 输出形状: (batch_size, 16, 98, 7)

        # 池化层1：仅在高度维度上进行池化
        x = F.max_pool2d(
            x, kernel_size=(2, 1), stride=(2, 1)
        )  # 输出形状: (batch_size, 16, 49, 7)

        # 卷积层2 -> ReLU
        x = F.relu(self.conv2(x))  # 输出形状: (batch_size, 32, 47, 5)

        # 池化层2：仅在高度维度上进行池化
        x = F.max_pool2d(
            x, kernel_size=(2, 1), stride=(2, 1)
        )  # 输出形状: (batch_size, 32, 23, 5)

        # 展平
        x = x.view(x.size(0), -1)  # 输出形状: (batch_size, 32*23*5=3680)

        # 全连接层1 -> ReLU
        x = F.relu(self.fc1(x))  # 输出形状: (batch_size, 128)

        # 丢弃
        x = self.dropout(x)

        # 全连接层2
        x = self.fc2(x)  # 输出形状: (batch_size, 3)

        return x


# 4. 定义损失函数和优化器
def define_model():
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer


# 5. 训练模型
def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs=20,
    save_path="simple_cnn_model.pth",
):
    """
    训练模型并在每个 epoch 后保存最佳模型。

    参数：
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        test_loader (DataLoader): 测试数据加载器。
        criterion: 损失函数。
        optimizer: 优化器。
        device: 训练设备（CPU 或 GPU）。
        epochs (int): 训练轮数。
        save_path (str): 最佳模型的保存路径。

    返回：
        list: 每个 epoch 的训练损失。
    """
    model.to(device)
    train_losses = []
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 评估模型
        accuracy = evaluate_model(model, test_loader, device)
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%"
        )

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f"保存模型 (Accuracy: {best_accuracy:.2f}%) 到 {save_path}")

    print("训练完成")
    return train_losses


# 6. 评估模型
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# 7. 加载并使用保存的模型
def load_best_model(model_path, device):
    """
    加载保存的最佳模型。

    参数：
        model_path (str): 保存的模型路径。
        device: 训练设备（CPU 或 GPU）。

    返回:
        nn.Module: 加载好的模型。
    """
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"最佳模型已从 {model_path} 加载")
    return model


def predict(model, sample, device):
    """
    使用模型对单个样本进行预测。

    参数：
        model (nn.Module): 已加载的模型。
        sample (numpy.ndarray): 单个样本数据，形状为 (100, 9)。
        device: 训练设备（CPU 或 GPU）。

    返回:
        int: 预测的标签类别（0, 1, 2）。
    """
    model.eval()
    with torch.no_grad():
        # 转换为张量并添加 batch 和 channel 维度
        sample_tensor = (
            torch.tensor(sample, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        # 前向传播
        output = model(sample_tensor)
        _, predicted = torch.max(output, 1)

        return predicted.item()


# 8. 主函数
def main():
    # 文件路径
    file_path = "merged_csv/BTCUSDT_1h_2022-12-08_to_2024-11-25_samples_with_labels.pkl"  #  .pkl 文件路径

    # 加载和处理数据
    samples, labels = load_and_process_pkl(file_path)
    if samples is None or labels is None:
        return

    # 映射标签
    labels = map_labels(labels)

    # 标准化样本
    samples = standardize_samples(samples)

    # 拆分数据
    X_train, X_test, y_train, y_test = split_train_test(samples, labels)

    # 创建数据集
    train_dataset = CustomMatrixDataset(X_train, y_train)
    test_dataset = CustomMatrixDataset(X_test, y_test)

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型、损失函数和优化器
    model, criterion, optimizer = define_model()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练模型
    epochs = 20
    save_path = "simple_cnn_model.pth"
    train_losses = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs=epochs,
        save_path=save_path,
    )

    # 加载模型进行最终评估
    model.load_state_dict(torch.load(save_path))
    accuracy = evaluate_model(model, test_loader, device)
    print(f"模型在测试集上的准确率: {accuracy:.2f}%")

    # 加载模型
    model = load_best_model(save_path, device)

    # 选择一个样本进行预测
    sample_index = 333  # 选择第一个样本
    sample = samples[sample_index]
    true_label = labels[sample_index]
    predicted_label = predict(model, sample, device)

    # 标签映射回原始标签
    label_mapping_reverse = {0: -1, 1: 0, 2: 1}
    predicted_label_original = label_mapping_reverse[predicted_label]

    print(f"真实标签: {true_label}, 预测标签: {predicted_label_original}")


if __name__ == "__main__":
    main()
