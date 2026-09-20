import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=28, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    return train_transform, val_test_transform


def get_dataloaders(
    data_root: str = "./data",
    dataset_name: str = "bloodmnist",
    batch_size: int = 128,
    num_workers: int = 2,
    device: str = "cuda",
):
    print("\n" + "=" * 70)
    print(f"LOADING {dataset_name.upper()}")
    print("=" * 70)

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    train_transform, val_test_transform = get_transforms()

    train_dataset = DataClass(
        split="train",
        root=data_root,
        transform=train_transform,
        download=True,
    )

    val_dataset = DataClass(
        split="val",
        root=data_root,
        transform=val_test_transform,
        download=True,
    )

    test_dataset = DataClass(
        split="test",
        root=data_root,
        transform=val_test_transform,
        download=True,
    )

    print(f"Dataset:            {dataset_name}")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples:    {len(test_dataset)}")

    pin_memory = device.startswith("cuda") and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
