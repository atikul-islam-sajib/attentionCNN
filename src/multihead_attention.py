import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from scaled_dot_product import scaled_dot_product_attention


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

            self.attention = scaled_dot_product_attention(
                query=self.query, key=self.key, value=self.value, channels=self.channels
            )

            assert (
                self.attention.size()
                == self.query.size()
                == self.key.size()
                == self.value.size()
            ), "Attention output must have the same size as QKV"

            self.attention = self.attention.view(
                self.attention.size(0),
                self.attention.size(1) * self.attention.size(2),
                self.attention.size(3) // self.channels,
                self.attention.size(3) // self.channels,
            )

            return self.layers(self.attention)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multihead attention layer for attentionCNN".capitalize()
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config()["attentionCNN"]["image_size"],
        help="Number of channels in the input tensor".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=config()["attentionCNN"]["nheads"],
        help="Number of heads in the multihead attention layer".capitalize(),
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=config()["attentionCNN"]["bias"],
        help="Whether to use bias in the multihead attention layer".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["dataloader"]["batch_size"]
    image_size = config()["dataloader"]["image_size"]

    attention = MultiHeadAttentionLayer(
        channels=args.channels, nheads=args.nheads, bias=args.bias
    )

    print(
        attention(
            torch.randn(batch_size, args.channels, args.image_size, args.image_size)
        ).size()
    )
