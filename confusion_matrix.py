import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from cnn import (
    CustomMatrixDataset,
    get_pkl_file_paths,
    load_multiple_pkls,
    map_labels,
    standardize_samples,
    load_model,
    confusion_matrix_report,
)


def main():
    # Directory containing .pkl files
    data_directory = "merged_csv"  # Update if different

    # Prompt user for interval input
    interval = input("Enter interval: ")
    pattern = f"*{interval}*.pkl"
    file_paths = get_pkl_file_paths(data_directory, pattern=pattern)
    if not file_paths:
        print(f"No .pkl files found in {data_directory} matching pattern {pattern}")
        return

    # Load data
    samples, labels = load_multiple_pkls(file_paths)
    if samples is None or labels is None:
        return

    print(f"Number of NaNs in samples: {np.isnan(samples).sum()}")
    print(f"Number of Infs in samples: {np.isinf(samples).sum()}")

    # Remove samples with NaN or Inf
    if np.isnan(samples).any() or np.isinf(samples).any():
        nan_indices = np.unique(np.argwhere(np.isnan(samples))[:, 0])
        inf_indices = np.unique(np.argwhere(np.isinf(samples))[:, 0])
        invalid_indices = np.unique(np.concatenate((nan_indices, inf_indices)))
        print(f"Removing {len(invalid_indices)} samples with NaN or Inf")
        samples = np.delete(samples, invalid_indices, axis=0)
        labels = np.delete(labels, invalid_indices, axis=0)

    # Map labels
    labels = map_labels(labels)

    # Standardize samples
    samples = standardize_samples(samples, scaler_path="scaler.pkl")

    print(f"Number of NaNs in standardized samples: {np.isnan(samples).sum()}")
    print(f"Number of Infs in standardized samples: {np.isinf(samples).sum()}")

    # Create datasets and dataloaders
    test_dataset = CustomMatrixDataset(samples, labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the trained model
    model_path = f"cnn_model_{interval}.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Generate confusion matrix and classification report
    confusion_matrix_report(model, test_loader, device, save_dir="plots")


if __name__ == "__main__":
    main()
