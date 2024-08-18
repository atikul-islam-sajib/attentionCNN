import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(IoULoss, self).__init__()

        self.name = "Iou Loss".title()
        self.smooth = smooth

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        if isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor):
            predicted = predicted.view(-1)
            target = target.view(-1)

            return 1 - (predicted * target).sum() / (
                predicted.sum()
                + target.sum()
                - (predicted * target).sum()
                + self.smooth
            )

        else:
            raise TypeError("Predicted and Target must be torch.Tensor".capitalize())


if __name__ == "__main__":
    loss = IoULoss()

    predicted = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
    target = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=torch.float32)

    print(loss(predicted, target))
