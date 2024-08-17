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

        self.layers = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            QKV = self.QKV(x)

            self.query, self.key, self.value = torch.chunk(input=QKV, chunks=3, dim=1)

            assert (
                self.query.size() == self.key.size() == self.value.size()
            ), "QKV must have the same size".capitalize()

            self.query = self.query.view(
                self.query.size(0),
                self.nheads,
                self.channels // self.nheads,
                self.query.size(2) * self.query.size(3),
            )

            self.key = self.key.view(
                self.key.size(0),
                self.nheads,
                self.channels // self.nheads,
                self.key.size(2) * self.key.size(3),
            )

            self.value = self.value.view(
                self.value.size(0),
                self.nheads,
                self.channels // self.nheads,
                self.value.size(2) * self.value.size(3),
            )

            return self.query


if __name__ == "__main__":
    attention = MultiHeadAttentionLayer(channels=128, nheads=8, bias=True)

    print(attention(torch.randn(16, 128, 128, 128)).size())
