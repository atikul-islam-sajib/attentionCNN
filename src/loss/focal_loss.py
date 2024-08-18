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
    parser = argparse.ArgumentParser(description="Focal loss for attentionCNN".title())
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for focal loss".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=2,
        help="Gamma parameter for focal loss".capitalize(),
    )

    args = parser.parse_args()

    loss = FocalLoss(alpha=args.alpha, gamma=args.gamma)

    predicted = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])

    assert (loss(predicted, target)).size() == torch.Size(
        []
    ), "Loss must be a scalar".capitalize()
