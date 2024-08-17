import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


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
                self.layers.append(nn.Dropout2d(p=self.dropout, inplace=True))

            self.channels = self.channels // self.upscale_factor**2
            self.out_channels = channels

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.model(x)

        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FeedForwardNeuralNetwork for the task of the attentionCNN".capitalize()
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=config()["attentionCNN"]["image_size"],
        help="Number of channels in the input image".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["attentionCNN"]["dropout"],
        help="Dropout rate for the network".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=config()["attentionCNN"]["activation"],
        help="Activation function for the network".capitalize(),
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=config()["attentionCNN"]["bias"],
        help="Whether to use bias in the network".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["dataloader"]["batch_size"]

    network = FeedForwardNeuralNetwork(channels=args.channels, dropout=args.dropout)

    assert network(
        torch.randn(batch_size, args.channels, args.channels, args.channels)
    ).size() == (
        batch_size,
        args.channels // 4,
        args.channels * 2,
        args.channels * 2,
    ), "Network output size is incorrect".capitalize()
