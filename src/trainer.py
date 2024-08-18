import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

from helper import helper


class Trainer:
    def __init__(
        self,
        model=None,
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
    ):
        self.model = model
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
