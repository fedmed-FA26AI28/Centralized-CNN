# Centralized CNN Baseline - BloodMNIST

A PyTorch implementation of a Centralized Convolutional Neural Network (CNN) trained and evaluated on the **BloodMNIST** dataset (from the [MedMNIST v2](https://medmnist.com/) biomedical benchmark).

This repository provides both a **modular command-line interface** for running headless training in the terminal and an **interactive Jupyter Notebook** with end-to-end visual analytics (dataset inspection, learning curves, hardware monitoring, and confusion matrix).

---

## 📁 Repository Structure

```text
Central/
├── config.py              # CLI arguments, default hyperparameters, and device setup
├── dataset.py             # Data augmentations, normalization, and DataLoader factories
├── model.py               # CNN architecture definition and parameter counting utility
├── utils.py               # Seed setting, resource monitoring (CPU/RAM/GPU), and evaluation
├── train.py               # Main CLI training orchestrator and metrics logger
├── nb.ipynb               # Interactive Jupyter Notebook with all visualizations
├── .gitignore             # Git exclusion rules for checkpoints, data, and cache
├── requirements.txt       # Project dependencies (unpinned)
└── README.md              # Project documentation
```

---

## ⚙️ Environment & Prerequisites

This project is tested with **Python 3.10** and **PyTorch 2.x** with CUDA GPU acceleration.

### Installation

Install dependencies using:
```powershell
pip install -r requirements.txt
```

### Using the Dedicated Virtual Environment

If using the workspace environment (`D:\FPT\KLTN\FED\.venv`):

**In PowerShell:**
```powershell
# Activate the virtual environment
& "D:\FPT\KLTN\FED\.venv\Scripts\Activate.ps1"
```

Or execute directly with the virtual environment Python binary:
```powershell
& "D:\FPT\KLTN\FED\.venv\Scripts\python.exe" train.py
```

---

## 🚀 Running via Terminal (`train.py`)

### 1. Default Run
Trains for 30 epochs with batch size 128 on CUDA (if available):
```powershell
python train.py
```

### 2. Custom Hyperparameters
You can override any parameter using command-line arguments:
```powershell
python train.py --epochs 20 --batch_size 64 --lr 0.0005 --num_workers 2 --device cuda
```

### 3. Available CLI Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--epochs` | `int` | `30` | Number of training epochs |
| `--batch_size` | `int` | `128` | Batch size for train/val/test data loaders |
| `--learning_rate` / `--lr` | `float` | `0.001` | Learning rate for Adam optimizer |
| `--num_classes` | `int` | `8` | Number of target classes |
| `--random_seed` / `--seed` | `int` | `42` | Random seed for reproducibility |
| `--num_workers` | `int` | `2` | Number of DataLoader worker subprocesses |
| `--device` | `str` | `cuda` or `cpu` | Execution hardware |
| `--data_root` | `str` | `./data` | Directory where BloodMNIST `.npz` is stored |
| `--model_path` | `str` | `bloodmnist_cnn.pt` | Path to save the best model weights |
| `--train_metrics_path` | `str` | `bloodmnist_training_metrics.csv` | Path to export epoch train/val metrics |
| `--resource_metrics_path` | `str` | `bloodmnist_resource_metrics.csv` | Path to export hardware utilization stats |
| `--final_results_path` | `str` | `bloodmnist_final_results.csv` | Path to export final test evaluation summary |

---

## 📊 Interactive Visualization Notebook (`nb.ipynb`)

[`nb.ipynb`](nb.ipynb) contains every phase of the pipeline split into standalone cells:
1. **Dataset Preview**: Visualizes 8 sample BloodMNIST images with denormalization and class labels.
2. **Model Architecture**: Displays layer composition and total parameter counts (20,104 parameters).
3. **Training Execution**: Real-time per-epoch feedback on loss, accuracy, precision, recall, and macro F1 score.
4. **Learning Curves Plot**: Multi-panel visualization comparing Training vs. Validation Loss, Accuracy, and Macro F1 over epochs.
5. **System Hardware Utilization Plot**: Multi-panel visualization tracking CPU %, RAM (MB), GPU Utilization %, and GPU Memory (MB) across training.
6. **Confusion Matrix & Classification Report**: Heatmap showing per-class true vs. predicted classifications on the test set.

---

## 💾 Output Artifacts

Upon completing training, the pipeline generates:
1. **`bloodmnist_cnn.pt`**: PyTorch checkpoint of the model weights that achieved the highest validation F1 score.
2. **`bloodmnist_training_metrics.csv`**: Detailed per-epoch breakdown (`epoch`, `train_loss`, `train_acc`, `train_f1`, `val_loss`, `val_acc`, `val_f1`, `epoch_time`).
3. **`bloodmnist_resource_metrics.csv`**: Hardware telemetry per epoch (`avg_cpu_percent`, `max_cpu_percent`, `avg_ram_mb`, `avg_gpu_percent`, `avg_gpu_memory_mb`).
4. **`bloodmnist_final_results.csv`**: Comprehensive one-row summary of the entire run including test set metrics and overall resource consumption.

---

## 💡 Troubleshooting

### VS Code "Could not register service worker / InvalidStateError"
If opening `.ipynb` files in VS Code produces a webview service worker error:
1. Press **`Ctrl + Shift + P`** in VS Code.
2. Select **`Developer: Reload Window`**.
3. Re-open the notebook.
