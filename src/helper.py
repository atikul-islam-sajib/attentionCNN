import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src/")

from utils import config
from attentionCNN import attentionCNN
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from loss.jaccard_loss import IoULoss
from loss.combo_loss import ComboLoss
from loss.tversky_loss import TverskyLoss
from loss.bce_loss import BinaryCrossEntropyLoss

def load_dataloader():
    if os.path.exists(config()["path"]["PROCESSED_PATH"]):
        train_dataloader = os.path.join(config()["path"]["PROCESSED_PATH"], "train_dataloader.pkl")
        valid_dataloader = os.path.join(config()["path"]["PROCESSED_PATH"], "valid_dataloader.pkl")
        test_dataloader = os.path.join(config()["path"]["PROCESSED_PATH"], "test_dataloader.pkl")
        
        return {
           "train_dataloader": train_dataloader,
           "valid_dataloader": valid_dataloader,
           "test_dataloader": test_dataloader, 
        }
        
    else:
        raise FileNotFoundError("dataloader cannot be imported from the helper method".capitalize())

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
        optimizer = optim.Adam(params=model.parameters(), lr = lr, betas=(beta1, beta2)) 
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
    elif loss == "combo":
        loss = ComboLoss(alpha=alpha, gamma=gamma, smooth=smooth)
    else:
        loss = BinaryCrossEntropyLoss(reduction="mean")
        
    return {
        "train_dataloader": dataloader["train_dataloader"],
        "valid_dataloader": dataloader["valid_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
        "model": model,
        "optimizer": optimizer,
        "loss": loss
    }

if __name__ == "__main__":
    pass        
    