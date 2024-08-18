import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src/")

from helper import helper
from utils import config, device_init
from attentionCNN import attentionCNN
from loss.focal_loss import FocalLoss
from loss.jaccard_loss import IoULoss
from loss.dice_loss import DiceLoss
from loss.tversky_loss import TverskyLoss
from loss.bce_loss import BinaryCrossEntropyLoss


class Trainer:
    def __init__(
        self,
        model=None,
        epochs: int = 100,
        lr: float = 0.0001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        momentum: float = 0.90,
        adam: bool = True,
        SGD: bool = False,
        loss="bce",
        smooth: float = 1e-4,
        alpha: float = 0.25,
        gamma: int = 2,
        device: str = "cuda",
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.adam = adam
        self.SGD = SGD
        self.loss = loss
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.device = device_init(
            device=device,
        )

        self.init = helper(
            model=self.model,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
            adam=self.adam,
            SGD=self.SGD,
            loss=self.loss,
            smooth=self.smooth,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]

        self.model = self.init["model"].to(self.device)

        self.optimizer = self.init["optimizer"]
        self.criterion = self.init["loss"]

        assert (
            self.init["train_dataloader"].__class__
        ) == torch.utils.data.dataloader.DataLoader, (
            "train_dataloader is not a dataloader".capitalize()
        )
        assert (
            self.init["valid_dataloader"].__class__
        ) == torch.utils.data.dataloader.DataLoader, (
            "valid_dataloader is not a dataloader".capitalize()
        )
        assert (
            self.init["test_dataloader"].__class__
        ) == torch.utils.data.dataloader.DataLoader, (
            "test_dataloader is not a dataloader".capitalize()
        )
        assert (
            self.init["model"].__class__
        ) == attentionCNN, "model is not a model".capitalize()
        assert (
            self.init["optimizer"].__class__
        ) == optim.Adam, "optimizer is not an optimizer".capitalize()
        assert (
            self.init["loss"].__class__
        ) == BinaryCrossEntropyLoss, "loss is not a loss function".capitalize()


if __name__ == "__main__":
    trainer = Trainer(epochs=1)
