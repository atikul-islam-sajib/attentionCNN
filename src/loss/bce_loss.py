import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(BinaryCrossEntropyLoss, self).__init__()

        self.reduction = reduction

        self.loss = nn.BCELoss(reduction=self.reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            return self.loss(pred, target)

        else:
            raise TypeError("pred and target must be torch.Tensor".capitalize())


if __name__ == "__main__":
    loss = BinaryCrossEntropyLoss()

    predicted = torch.tensor([1.0, 0.0, 1.0, 0.0])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    print(loss(predicted, target))