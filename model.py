import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()

        # Conv Block 1: Input 3 x 28 x 28 -> Output 32 x 28 x 28
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Conv Block 2: Input 32 x 28 x 28 -> Output 64 x 28 x 28
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Max Pooling: 64 x 28 x 28 -> 64 x 14 x 14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling: 64 x 14 x 14 -> 64 x 1 x 1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer: 64 -> num_classes
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module):
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_parameters, trainable_parameters
