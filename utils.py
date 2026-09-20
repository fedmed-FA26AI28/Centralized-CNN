import random
import subprocess
import time
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_gpu_metrics():
    if not torch.cuda.is_available():
        return 0.0, 0.0

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        first_line = result.strip().splitlines()[0]
        values = first_line.split(",")
        gpu_utilization = float(values[0].strip())
        gpu_memory_mb = float(values[1].strip())
        return gpu_utilization, gpu_memory_mb
    except Exception:
        # Fallback to PyTorch CUDA memory tracking if nvidia-smi fails
        try:
            allocated_mb = torch.cuda.memory_allocated() / (1024**2)
            return 0.0, float(allocated_mb)
        except Exception:
            return 0.0, 0.0


def get_resource_metrics():
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_mb = psutil.virtual_memory().used / (1024**2)
    gpu_percent, gpu_memory_mb = get_gpu_metrics()

    return {
        "cpu_percent": cpu_percent,
        "ram_mb": ram_mb,
        "gpu_percent": gpu_percent,
        "gpu_memory_mb": gpu_memory_mb,
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    elapsed_time = time.time() - start_time

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time": elapsed_time,
    }
