import sys
import torch
import argparse
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, channels: int = 128, nheads: int = 8, bias: bool = True):
        super(MultiHeadAttentionLayer, self).__init__()

        self.channels = channels
        self.nheads = nheads
        self.bias = bias

        assert (
            self.channels % self.nheads == 0
        ), "Channels must be divisible by number of heads".capitalize()

        self.kernel_size = 1
        self.stride = 1
        self.padding = 0

        self.QKV = nn.Conv2d(
            in_channels=self.channels,
            out_channels=3 * self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
