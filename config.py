import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized CNN Training on BloodMNIST"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for dataset storage",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bloodmnist",
        help="MedMNIST dataset name",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="Number of classification classes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Input batch size for training/testing",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--random_seed",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="bloodmnist_cnn.pt",
        help="File path to save the best model weights",
    )
    parser.add_argument(
        "--train_metrics_path",
        type=str,
        default="bloodmnist_training_metrics.csv",
        help="File path to save epoch training/validation metrics",
    )
    parser.add_argument(
        "--resource_metrics_path",
        type=str,
        default="bloodmnist_resource_metrics.csv",
        help="File path to save hardware resource utilization metrics",
    )
    parser.add_argument(
        "--final_results_path",
        type=str,
        default="bloodmnist_final_results.csv",
        help="File path to save overall final evaluation results",
    )

    return parser.parse_args()


def print_device_info(device_str: str):
    device = torch.device(device_str)
    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)
    print(f"Device: {device}")

    if device.type == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("GPU not in use. Running on CPU.")
    print("=" * 70)
    return device
