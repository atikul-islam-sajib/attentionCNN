import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-3):
        super(DiceLoss, self).__init__()

        self.name = "DiceLoss".title()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            pred = pred.contiguous().view(-1)
            target = target.contiguous().view(-1)

            return (pred * target).sum() / (pred.sum() + target.sum() + self.smooth)

        else:
            raise TypeError("pred and target must be torch.Tensor".capitalize())


if __name__ == "__main__":
    loss = DiceLoss()

    predicted = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])
    target = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])

    assert (
        type(loss(predicted, target)) == torch.Tensor
    ), "Loss must be a torch.Tensor".capitalize()
