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

            dice_coefficient = (2 * (pred * target).sum()) / (
                pred.sum() + target.sum() + self.smooth
            )

            return 1 - dice_coefficient

        else:
            raise TypeError("pred and target must be torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiceLoss for attentionCNN".title())
    parser.add_argument(
        "--smooth", type=float, default=1e-3, help="smooth parameter for DiceLoss"
    )

    args = parser.parse_args()

    loss = DiceLoss(smooth=args.smooth)

    predicted = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])
    target = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])

    assert (
        type(loss(predicted, target)) == torch.Tensor
    ), "Loss must be a torch.Tensor".capitalize()
