import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from dice_loss import DiceLoss
from focal_loss import FocalLoss
from bce_loss import BinaryCrossEntropyLoss


class ComboLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: int = 2,
        smooth: float = 1e-4,
        reduction: str = "mean",
    ):
        super(ComboLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

        self.dice_loss = DiceLoss(smooth=self.smooth)
        self.focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        self.bce_loss = BinaryCrossEntropyLoss(reduction=self.reduction)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        if isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor):

            predicted = predicted.contiguous().view(-1)
            target = target.contiguous().view(-1)

            return (
                self.dice_loss(predicted, target)
                + self.focal_loss(predicted, target)
                + self.bce_loss(predicted, target)
            ).mean()
        else:
            raise TypeError("Predicted and target must be torch.Tensor".capitalize())


if __name__ == "__main__":
    loss = ComboLoss(alpha=0.25, gamma=2, smooth=1e-4, reduction="mean")

    predicted = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float)
    target = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float)

    assert (
        type(loss(predicted, target)) == torch.Tensor
    ), "Loss must be a torch.Tensor".capitalize()
