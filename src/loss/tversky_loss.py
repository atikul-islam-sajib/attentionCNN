import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class TverskyLoss(nn.Module):
    def __init__(self, name: str = "TveskyLoss"):
        super(TverskyLoss, self).__init__()

        self.name = name

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        if isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor):
            TP = torch.sum(predicted * target)
            FP = torch.sum(predicted * (1 - target))
            FN = torch.sum((1 - predicted) * target)

            return 1 - (TP / (TP + 0.5 * (FP + FN)))

        else:
            raise TypeError("Predicted and target must be torch.Tensor".capitalize())


if __name__ == "__main__":
    loss = TverskyLoss()

    predicted = torch.tensor([1.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert (loss(predicted, target)).size() == torch.Size(
        []
    ), "Loss must be a scalar".capitalize()
