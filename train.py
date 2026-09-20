import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import parse_args, print_device_info
from dataset import get_dataloaders
from model import CNN, count_parameters
from utils import evaluate, get_resource_metrics, set_seed


def main():
    args = parse_args()

    # 1. Device and reproducibility setup
    device = print_device_info(args.device)
    set_seed(args.random_seed)

    # 2. DataLoaders
    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    # 3. Model setup
    model = CNN(num_classes=args.num_classes).to(device)

    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    print(model)

    total_parameters, trainable_parameters = count_parameters(model)
    print(f"\nTotal parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 5. Training loop
    training_history = []
    resource_history = []
    best_val_f1 = -1.0

    print("\n" + "=" * 70)
    print(f"STARTING TRAINING ({args.epochs} EPOCHS)")
    print("=" * 70)

    total_training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        all_train_predictions = []
        all_train_labels = []

        epoch_cpu = []
        epoch_ram = []
        epoch_gpu = []
        epoch_gpu_memory = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Batch stats
            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)

            all_train_predictions.extend(predictions.detach().cpu().numpy())
            all_train_labels.extend(labels.detach().cpu().numpy())

            # Resource monitoring
            resources = get_resource_metrics()
            epoch_cpu.append(resources["cpu_percent"])
            epoch_ram.append(resources["ram_mb"])
            epoch_gpu.append(resources["gpu_percent"])
            epoch_gpu_memory.append(resources["gpu_memory_mb"])

        # Epoch training metrics
        train_loss = running_loss / len(train_loader.dataset)
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
        train_precision = precision_score(
            all_train_labels,
            all_train_predictions,
            average="macro",
            zero_division=0,
        )
        train_recall = recall_score(
            all_train_labels,
            all_train_predictions,
            average="macro",
            zero_division=0,
        )
        train_f1 = f1_score(
            all_train_labels,
            all_train_predictions,
            average="macro",
            zero_division=0,
        )

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        avg_cpu = float(np.mean(epoch_cpu)) if epoch_cpu else 0.0
        max_cpu = float(np.max(epoch_cpu)) if epoch_cpu else 0.0
        avg_ram = float(np.mean(epoch_ram)) if epoch_ram else 0.0
        max_ram = float(np.max(epoch_ram)) if epoch_ram else 0.0
        avg_gpu = float(np.mean(epoch_gpu)) if epoch_gpu else 0.0
        max_gpu = float(np.max(epoch_gpu)) if epoch_gpu else 0.0
        avg_gpu_memory = (
            float(np.mean(epoch_gpu_memory)) if epoch_gpu_memory else 0.0
        )
        max_gpu_memory = (
            float(np.max(epoch_gpu_memory)) if epoch_gpu_memory else 0.0
        )

        # Store histories
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "epoch_time_seconds": epoch_time,
        })

        resource_history.append({
            "epoch": epoch,
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "avg_ram_mb": avg_ram,
            "max_ram_mb": max_ram,
            "avg_gpu_percent": avg_gpu,
            "max_gpu_percent": max_gpu,
            "avg_gpu_memory_mb": avg_gpu_memory,
            "max_gpu_memory_mb": max_gpu_memory,
            "epoch_time_seconds": epoch_time,
        })

        # Best model checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), args.model_path)
            best_marker = " <-- BEST"
        else:
            best_marker = ""

        # Progress reporting
        print(f"\nEpoch [{epoch:02d}/{args.epochs}] | Time: {epoch_time:.2f}s")
        print(
            f"Train | Loss: {train_loss:.4f} | Acc: {train_accuracy:.4f} | "
            f"Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}"
        )
        print(
            f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}{best_marker}"
        )
        print(
            f"CPU: Avg {avg_cpu:.1f}%, Max {max_cpu:.1f}% | "
            f"RAM: Avg {avg_ram:.1f}MB, Max {max_ram:.1f}MB | "
            f"GPU: Avg {avg_gpu:.1f}%, Max {max_gpu:.1f}% | "
            f"GPU Mem: Avg {avg_gpu_memory:.1f}MB, Max {max_gpu_memory:.1f}MB"
        )

    total_training_time = time.time() - total_training_start

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best validation F1:  {best_val_f1:.4f}")
    print(f"Best model saved to: {args.model_path}")

    # 6. Final Test Evaluation using best checkpoint
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (Best Checkpoint)")
    print("=" * 70)

    if os.path.exists(args.model_path):
        print(f"Loading weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    test_start = time.time()
    test_metrics = evaluate(model, test_loader, criterion, device)
    test_time = time.time() - test_start

    print(f"Test Loss:      {test_metrics['loss']:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-score:  {test_metrics['f1']:.4f}")
    print(f"Test Time:      {test_time:.2f} seconds")

    # 7. Aggregate overall resource metrics
    all_resource_data = pd.DataFrame(resource_history)
    overall_resource_metrics = {
        "avg_cpu_percent": float(all_resource_data["avg_cpu_percent"].mean()),
        "max_cpu_percent": float(all_resource_data["max_cpu_percent"].max()),
        "avg_ram_mb": float(all_resource_data["avg_ram_mb"].mean()),
        "max_ram_mb": float(all_resource_data["max_ram_mb"].max()),
        "avg_gpu_percent": float(all_resource_data["avg_gpu_percent"].mean()),
        "max_gpu_percent": float(all_resource_data["max_gpu_percent"].max()),
        "avg_gpu_memory_mb": float(
            all_resource_data["avg_gpu_memory_mb"].mean()
        ),
        "max_gpu_memory_mb": float(
            all_resource_data["max_gpu_memory_mb"].max()
        ),
        "total_training_time_seconds": total_training_time,
    }

    # 8. Save output metrics
    final_results = {
        "model": "Centralized CNN",
        "dataset": args.dataset_name,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_time_seconds": test_time,
        "total_training_time_seconds": total_training_time,
        "avg_cpu_percent": overall_resource_metrics["avg_cpu_percent"],
        "max_cpu_percent": overall_resource_metrics["max_cpu_percent"],
        "avg_ram_mb": overall_resource_metrics["avg_ram_mb"],
        "max_ram_mb": overall_resource_metrics["max_ram_mb"],
        "avg_gpu_percent": overall_resource_metrics["avg_gpu_percent"],
        "max_gpu_percent": overall_resource_metrics["max_gpu_percent"],
        "avg_gpu_memory_mb": overall_resource_metrics["avg_gpu_memory_mb"],
        "max_gpu_memory_mb": overall_resource_metrics["max_gpu_memory_mb"],
    }

    training_df = pd.DataFrame(training_history)
    resource_df = pd.DataFrame(resource_history)
    final_df = pd.DataFrame([final_results])

    training_df.to_csv(args.train_metrics_path, index=False)
    resource_df.to_csv(args.resource_metrics_path, index=False)
    final_df.to_csv(args.final_results_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("Model:            Centralized CNN")
    print(f"Dataset:          {args.dataset_name}")
    print(f"Train samples:    {len(train_dataset)}")
    print(f"Validation:       {len(val_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")
    print(f"Test Loss:        {test_metrics['loss']:.4f}")
    print(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Test Precision:   {test_metrics['precision']:.4f}")
    print(f"Test Recall:      {test_metrics['recall']:.4f}")
    print(f"Test F1:          {test_metrics['f1']:.4f}")
    print(f"Training Time:    {total_training_time:.2f}s")
    print(f"Max CPU:          {overall_resource_metrics['max_cpu_percent']:.2f}%")
    print(f"Max RAM:          {overall_resource_metrics['max_ram_mb']:.2f} MB")
    print(f"Max GPU:          {overall_resource_metrics['max_gpu_percent']:.2f}%")
    print(
        f"Max GPU Memory:   {overall_resource_metrics['max_gpu_memory_mb']:.2f} MB"
    )

    print("\nSaved files:")
    print(f"1. {args.model_path}")
    print(f"2. {args.train_metrics_path}")
    print(f"3. {args.resource_metrics_path}")
    print(f"4. {args.final_results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
