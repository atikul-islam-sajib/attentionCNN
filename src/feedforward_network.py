import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.channels = channels
        self.dropout = dropout
        self.activation = activation
        self.bias = bias

        self.kernel_size = 1
        self.stride = 1
        self.padding = 0
        self.upscale_factor = 2

        if activation == "leaky_relu":
            self.activation == nn.LeakyReLU(inplace=True, negative_slope=0.2)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)

        self.layers = []

        for index in range(2):
            self.layers.append(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=self.bias,
                )
            )
            if index == 0:
                self.layers.append(nn.PixelShuffle(upscale_factor=self.upscale_factor))
                self.layers.append(self.activation)

            self.channels = self.channels // 4
            self.out_channels = channels

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.model(x)


if __name__ == "__main__":
    network = FeedForwardNeuralNetwork(channels=256, dropout=0.1)

    print(network(torch.randn(16, 256, 256, 256)).size())
