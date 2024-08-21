import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class MeanSquaredLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(MeanSquaredLoss, self).__init__()

        self.reduction = reduction

        self.loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            return self.loss(pred, target)

        else:
            raise TypeError("pred and target must be torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCELoss for attentionCNN".title())
    parser.add_argument(
        "--reduction", type=str, default="mean", help="mean or sum or none".capitalize()
    )

    loss = MeanSquaredLoss()

    predicted = torch.tensor([1.0, 0.0, 1.0, 0.0])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    assert loss(predicted, target).size() == torch.Size(
        []
    ), "BCELoss is not working".capitalize()
