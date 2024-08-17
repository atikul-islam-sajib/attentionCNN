import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self, channels: int = 128, dropout: float = 0.1, activation: str = "relu"
    ):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.channels = channels
        self.dropout = dropout
        self.activation = activation

        if activation == "leaky_relu":
            self.activation == nn.LeakyReLU(inplace=True, negative_slope=0.2)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)
