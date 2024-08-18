import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: int = 2):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        if isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor):
            predicted = predicted.view(-1)
            target = target.view(-1)

            BCELoss = nn.BCELoss()(predicted, target)
            pt = torch.exp(-BCELoss)

            return (self.alpha * (1 - pt) ** self.gamma * BCELoss).mean()
        else:
            raise TypeError("Predicted and target must be torch.Tensor".capitalize())


if __name__ == "__main__":
    loss = FocalLoss(alpha=0.25, gamma=2)

    predicted = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])

    assert (loss(predicted, target)).size() == torch.Size(
        []
    ), "Loss must be a scalar".capitalize()
