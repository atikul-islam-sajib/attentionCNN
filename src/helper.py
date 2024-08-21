import os
import sys
import torch
import torch.optim as optim

sys.path.append("./src/")

from utils import config, load
from attentionCNN import attentionCNN
from loss.focal_loss import FocalLoss
from loss.jaccard_loss import IoULoss
from loss.dice_loss import DiceLoss
from loss.tversky_loss import TverskyLoss
from loss.bce_loss import BinaryCrossEntropyLoss
from loss.mse_loss import MeanSquaredLoss


def load_dataloader():
    if os.path.exists(config()["path"]["PROCESSED_PATH"]):
        train_dataloader = os.path.join(
            config()["path"]["PROCESSED_PATH"], "train_dataloader.pkl"
        )
        valid_dataloader = os.path.join(
            config()["path"]["PROCESSED_PATH"], "valid_dataloader.pkl"
        )
        test_dataloader = os.path.join(
            config()["path"]["PROCESSED_PATH"], "test_dataloader.pkl"
        )

        return {
            "train_dataloader": load(filename=train_dataloader),
            "valid_dataloader": load(filename=valid_dataloader),
            "test_dataloader": load(filename=test_dataloader),
        }

    else:
        raise FileNotFoundError(
            "dataloader cannot be imported from the helper method".capitalize()
        )


def helper(**kwargs):
    model = kwargs["model"]
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    loss = kwargs["loss"]
    smooth = kwargs["smooth"]
    alpha = kwargs["alpha"]
    gamma = kwargs["gamma"]

    if model is None:
        model = attentionCNN(
            image_channels=config()["attentionCNN"]["image_channels"],
            image_size=config()["attentionCNN"]["image_size"],
            nheads=config()["attentionCNN"]["nheads"],
            dropout=config()["attentionCNN"]["dropout"],
            num_layers=config()["attentionCNN"]["num_layers"],
            activation=config()["attentionCNN"]["activation"],
            bias=True,
        )

    if adam:
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=config()["Trainer"]["weight_decay"],
        )
    elif SGD:
        optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)

    try:
        dataloader = load_dataloader()
    except FileNotFoundError as e:
        print("An error is occurred in the file ", e)
    except Exception as e:
        print("An error is occurred in the file ", e)

    if loss == "dice":
        loss = DiceLoss(smooth=smooth)
    elif loss == "focal":
        loss = FocalLoss(alpha=alpha, gamma=gamma)
    elif loss == "IoU":
        loss = IoULoss(smooth=smooth)
    elif loss == "tversky":
        loss = TverskyLoss()
    elif loss == "mse":
        loss = MeanSquaredLoss()
    else:
        loss = BinaryCrossEntropyLoss(reduction="mean")

    return {
        "train_dataloader": dataloader["train_dataloader"],
        "valid_dataloader": dataloader["valid_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
        "model": model,
        "optimizer": optimizer,
        "loss": loss,
    }


if __name__ == "__main__":
    init = helper(
        model=None,
        lr=0.001,
        momentum=0.9,
        loss="dice",
        alpha=0.25,
        gamma=2,
        smooth=1e-4,
        beta1=0.5,
        beta2=0.999,
        adam=True,
        SGD=False,
    )

    assert (
        init["train_dataloader"].__class__
    ) == torch.utils.data.dataloader.DataLoader, (
        "train_dataloader is not a dataloader".capitalize()
    )
    assert (
        init["valid_dataloader"].__class__
    ) == torch.utils.data.dataloader.DataLoader, (
        "valid_dataloader is not a dataloader".capitalize()
    )
    assert (
        init["test_dataloader"].__class__
    ) == torch.utils.data.dataloader.DataLoader, (
        "test_dataloader is not a dataloader".capitalize()
    )
    assert (
        init["model"].__class__
    ) == attentionCNN, "model is not a model".capitalize()
    assert (
        init["optimizer"].__class__
    ) == optim.Adam, "optimizer is not an optimizer".capitalize()
    assert (
        init["loss"].__class__
    ) == DiceLoss, "loss is not a loss function".capitalize()
