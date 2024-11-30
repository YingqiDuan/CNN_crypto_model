import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


# 1. Data Loading and Preprocessing
def load_multiple_pkls(file_paths):
    """
    读取多个 .pkl 文件，提取 'samples' 和 'labels'，合并所有数据。

    参数：
        file_paths (list): .pkl 文件的路径列表。

    返回:
        tuple: 合并后的 samples 和 labels。
            - samples (numpy.ndarray): 形状为 (总样本数量, 100, 9) 的数组。
            - labels (numpy.ndarray): 形状为 (总样本数量,) 的数组。
    """
    all_samples = []
    all_labels = []

    for file_path in file_paths:
        try:
            with open(file_path, "rb") as file:
                data = pickle.load(file)

            if not isinstance(data, dict):
                raise TypeError(f"文件 {file_path} 中的数据不是字典类型。")

            if "samples" not in data or "labels" not in data:
                raise KeyError(f"文件 {file_path} 中缺少 'samples' 或 'labels' 键。")

            samples = data["samples"]
            labels = data["labels"]

            samples = np.array(samples)
            labels = np.array(labels)

            if samples.ndim != 3 or samples.shape[1] != 100 or samples.shape[2] != 9:
                raise ValueError(
                    f"文件 {file_path} 中每个样本的矩阵形状应为 (100, 9)。当前形状: {samples.shape}"
                )

            all_samples.append(samples)
            all_labels.append(labels)

            # print(
            #     f"成功加载文件: {file_path}, Samples: {samples.shape}, Labels: {labels.shape}"
            # )

        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
        except pickle.UnpicklingError:
            print(f"无法解序列化文件: {file_path}")
        except KeyError as e:
            print(f"文件 {file_path} 中缺少必要的键: {e}")
        except TypeError as e:
            print(f"数据类型错误 in {file_path}: {e}")
        except ValueError as e:
            print(f"值错误 in {file_path}: {e}")
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")

    if not all_samples:
        print("没有成功加载任何数据。")
        return None, None

    # 合并所有样本和标签
    combined_samples = np.concatenate(all_samples, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    print(f"合并后的 Samples 形状: {combined_samples.shape}")
    print(f"合并后的 Labels 形状: {combined_labels.shape}")

    return combined_samples, combined_labels


def get_pkl_file_paths(directory, pattern="*.pkl"):
    """
    获取指定目录下所有匹配模式的 .pkl 文件路径。

    参数：
        directory (str): 目录路径。
        pattern (str): 文件匹配模式，默认是 "*.pkl"。

    返回:
        list: 匹配的文件路径列表。
    """
    search_pattern = os.path.join(directory, pattern)
    file_paths = glob.glob(search_pattern)
    return file_paths


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
        """
        自定义数据集。

        参数：
            samples (numpy.ndarray): 样本数据，形状为 (样本数量, 100, 9)。
            labels (numpy.ndarray): 标签数据，形状为 (样本数量,)。
            transform (callable, optional): 应用于样本的可选转换。
        """
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

        # 添加 channel 维度，变为 (1, 100, 9)
        sample = sample.unsqueeze(0)

        return sample, label


def split_train_val_test(samples, labels, val_size=0.1, test_size=0.1, random_state=42):
    """
    将数据集拆分为训练集、验证集和测试集。

    参数：
        samples (numpy.ndarray): 样本数据。
        labels (numpy.ndarray): 标签数据。
        val_size (float): 验证集所占比例。
        test_size (float): 测试集所占比例。
        random_state (int): 随机种子。

    返回:
        tuple: 拆分后的训练集、验证集和测试集。
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        samples,
        labels,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=labels,
    )
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# 3. 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1：输入通道=1，输出通道=32，卷积核大小=3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化
        # 卷积层2：输入通道=32，输出通道=64，卷积核大小=3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)  # 批归一化
        # 卷积层3：输入通道=64，输出通道=128，卷积核大小=3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)  # 批归一化
        # 丢弃层
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层
        self.fc1 = nn.Linear(128 * 10 * 3, 256)  # 根据新的输出尺寸调整
        self.fc2 = nn.Linear(256, 3)  # 3 个类别

    def forward(self, x):
        # 卷积层1 -> 批归一化 -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))  # 输出形状: (batch_size, 32, 98, 7)

        # 池化层1：仅在高度维度上进行池化
        x = F.max_pool2d(
            x, kernel_size=(2, 1), stride=(2, 1)
        )  # 输出形状: (batch_size, 32, 49, 7)

        # 卷积层2 -> 批归一化 -> ReLU
        x = F.relu(self.bn2(self.conv2(x)))  # 输出形状: (batch_size, 64, 47, 5)

        # 池化层2：仅在高度维度上进行池化
        x = F.max_pool2d(
            x, kernel_size=(2, 1), stride=(2, 1)
        )  # 输出形状: (batch_size, 64, 23, 5)

        # 卷积层3 -> 批归一化 -> ReLU
        x = F.relu(self.bn3(self.conv3(x)))  # 输出形状: (batch_size, 128, 21, 3)

        # 池化层3：仅在高度维度上进行池化
        x = F.max_pool2d(
            x, kernel_size=(2, 1), stride=(2, 1)
        )  # 输出形状: (batch_size, 128, 10, 3)

        # 展平
        x = x.view(x.size(0), -1)  # 输出形状: (batch_size, 128*10*3)

        # 全连接层1 -> ReLU
        x = F.relu(self.fc1(x))  # 输出形状: (batch_size, 256)

        # 丢弃
        x = self.dropout(x)

        # 全连接层2
        x = self.fc2(x)  # 输出形状: (batch_size, 3)

        return x


# 4. 定义损失函数和优化器
def define_model(y_train, device):
    model = CNN()
    # 计算类权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # 定义损失函数，使用类权重
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # 定义优化器，添加权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    return model, criterion, optimizer


# 5. 训练模型
def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs=20,
    save_path="cnn_model.pth",
    patience=5,
):
    """
    训练模型并在每个 epoch 后保存最佳模型。

    参数：
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        val_loader (DataLoader): 验证数据加载器。
        test_loader (DataLoader): 测试数据加载器。
        criterion: 损失函数。
        optimizer: 优化器。
        device: 训练设备（CPU 或 GPU）。
        epochs (int): 训练轮数。
        save_path (str): 模型的保存路径。
        patience (int): 早停的耐心参数。

    返回:
        list: 每个 epoch 的训练损失。
    """
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_accuracy = 0.0
    epochs_no_improve = 0

    # Initialize TensorBoard writer
    # in terminal: tensorboard --logdir=tensorboard
    writer = SummaryWriter(log_dir="tensorboard")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
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

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # 调整学习率
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"保存模型 (Accuracy: {best_accuracy:.2f}%) 到 {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"没有改善的epoch数量: {epochs_no_improve}")

        # 早停
        if epochs_no_improve >= patience:
            print("早停触发，停止训练。")
            break

    print("训练完成")
    return history


# 6. 评估模型
def evaluate_model(model, test_loader, device):
    """
    评估模型。

    参数：
        model (nn.Module): 已训练的模型。
        data_loader (DataLoader): 测试数据加载器。
        device: 训练设备（CPU 或 GPU）。

    返回:
        float: 数据集的准确率。
    """
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


def evaluate_model_with_probs(model, data_loader, device):
    """
    评估模型并返回真实标签和预测的概率分数。

    参数：
        model (nn.Module): 已训练的模型。
        data_loader (DataLoader): 数据加载器。
        device: 训练设备（CPU 或 GPU）。

    返回:
        tuple: 真实标签列表和预测概率列表。
    """
    model.to(device)
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_probs


def plot_roc_curves(y_true, y_probs, num_classes=3, save_dir="plots"):
    """
    绘制每个类别的ROC曲线并保存。

    参数：
        y_true (list or array): 真实的标签。
        y_probs (list or array): 预测的概率分数。
        num_classes (int): 类别数量。
        save_dir (str): 保存图像的目录路径。

    返回:
        None
    """

    os.makedirs(save_dir, exist_ok=True)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_probs = np.array(y_probs)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = ["aqua", "darkorange", "cornflowerblue"]
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:0.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    plt.close()
    print(f"ROC curves saved to {save_dir}/roc_curves.png")


# 7. 生成混淆矩阵和分类报告
def confusion_matrix_report(model, data_loader, device, save_dir="plots"):
    """
    生成混淆矩阵和分类报告，并将混淆矩阵绘制为热图保存。

    参数：
        model (nn.Module): 已训练的模型。
        data_loader (DataLoader): 测试数据加载器。
        device: 训练设备（CPU 或 GPU）。
        save_dir (str): 保存图像的目录路径。

    返回:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=["-1", "0", "1"])
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)

    # 绘制混淆矩阵热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["-1", "0", "1"],
        yticklabels=["-1", "0", "1"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix heatmap saved to {save_dir}/confusion_matrix.png")

    # save the classification report to a text file
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(cr)
    print(f"Classification report saved to {save_dir}/classification_report.txt")


# 8. 绘制学习曲线
def plot_learning_curves(history, save_dir="plots"):
    """
    绘制训练和验证的损失与准确率曲线，并保存为图像文件。

    参数：
        history (dict): 包含训练和验证损失与准确率的字典。
        save_dir (str): 保存图像的目录路径。

    返回:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], "bo-", label="train loss")
    plt.plot(epochs, history["val_loss"], "ro-", label="val loss")
    plt.title("train and val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_accuracy"], "bo-", label="train_accuracy")
    plt.plot(epochs, history["val_accuracy"], "ro-", label="val_accuracy")
    plt.title("train and val accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()


# 9. 加载并使用保存的模型
def load_model(model_path, device):
    """
    加载保存的模型。

    参数：
        model_path (str): 保存的模型路径。
        device: 训练设备（CPU 或 GPU）。

    返回:
        nn.Module: 加载好的模型。
    """
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已从 {model_path} 加载")
    return model


def predict(model, sample, device):
    """
    使用模型对单个样本进行预测。

    参数：
        model (nn.Module): 已加载的模型。
        sample (numpy.ndarray): 单个样本数据，形状为 (100, 9)。
        device: 训练设备（CPU 或 GPU）。

    返回:
        int: 预测的标签类别（-1, 0, 1）。
    """
    model.eval()
    # 转换为张量并添加 batch 和 channel 维度
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():

        # 前向传播
        output = model(sample_tensor)
        _, predicted_label = torch.max(output, 1)
        predicted_label = predicted_label.cpu().numpy()[0]

        # 标签映射回原始标签
        label_mapping_reverse = {0: -1, 1: 0, 2: 1}

        return label_mapping_reverse[predicted_label.item()]


# 10. 主函数
def main():
    # 多个 .pkl 文件所在的目录路径
    data_directory = "merged_csv"  # 替换为包含多个 .pkl 文件的目录路径

    # 获取所有 .pkl 文件路径
    file_paths = get_pkl_file_paths(data_directory)
    if not file_paths:
        print(f"在目录 {data_directory} 中未找到任何 .pkl 文件。")
        return

    # 加载和处理数据
    samples, labels = load_multiple_pkls(file_paths)
    if samples is None or labels is None:
        return

    print(f"Number of NaNs in samples: {np.isnan(samples).sum()}")
    print(f"Number of Infs in samples: {np.isinf(samples).sum()}")

    # 如果存在 NaN 或 Inf，删除这些样本
    if np.isnan(samples).any() or np.isinf(samples).any():
        nan_indices = np.unique(np.argwhere(np.isnan(samples))[:, 0])
        inf_indices = np.unique(np.argwhere(np.isinf(samples))[:, 0])
        invalid_indices = np.unique(np.concatenate((nan_indices, inf_indices)))
        print(f"删除包含 NaN 或 Inf 的样本数量: {len(invalid_indices)}")
        samples = np.delete(samples, invalid_indices, axis=0)
        labels = np.delete(labels, invalid_indices, axis=0)

    # 映射标签
    labels = map_labels(labels)

    # 标准化样本
    samples = standardize_samples(samples)

    print(f"Number of NaNs in standardized samples: {np.isnan(samples).sum()}")
    print(f"Number of Infs in standardized samples: {np.isinf(samples).sum()}")

    # 拆分数据
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        samples, labels
    )
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    # 创建数据集
    train_dataset = CustomMatrixDataset(X_train, y_train)
    val_dataset = CustomMatrixDataset(X_val, y_val)
    test_dataset = CustomMatrixDataset(X_test, y_test)

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型、损失函数和优化器
    model, criterion, optimizer = define_model(y_train, device)

    # 训练模型
    epochs = 100
    save_path = "cnn_model.pth"
    history = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs=epochs,
        save_path=save_path,
        patience=10,
    )

    # 绘制学习曲线
    plot_learning_curves(history)

    # 加载模型进行最终评估
    model.load_state_dict(torch.load(save_path, map_location=device))
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {accuracy:.2f}%")

    # 生成混淆矩阵和分类报告
    confusion_matrix_report(model, test_loader, device)

    # Generate ROC curves
    y_true, y_probs = evaluate_model_with_probs(model, test_loader, device)
    plot_roc_curves(y_true, y_probs, num_classes=3)


if __name__ == "__main__":
    main()
