import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int = 128, out_channels: int = 256, batch_norm: bool = True
    ):
        super(EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batchnorm = batch_norm

        self.kernel_size = 4
        self.stride_size = 2
        self.padding_size = 1

        self.layers = list()

        self.encoder_block = self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                    padding=self.padding_size,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size - 1,
                    stride=self.stride_size // self.stride_size,
                    padding=self.padding_size,
                ),
            )
        )

        if self.use_batchnorm:
            self.layers.append(nn.BatchNorm2d(num_features=self.out_channels))

        self.encoder_block = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.encoder_block(x)

        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder Block for attentionCNN".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=128,
        help="Number of input channels for the encoder block".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=256,
        help="Number of output channels for the encoder block".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels

    batch_size = config()["dataloader"]["batch_size"]
    image_size = config()["attentionCNN"]["image_size"]

    layers = []

    for index in range(3):
        layers.append(
            EncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                batch_norm=False if index == 3 - 1 else True,
            )
        )

        in_channels = out_channels
        out_channels = out_channels * 2

    model = nn.Sequential(*layers)

    assert (
        model(torch.randn(batch_size, args.in_channels, image_size, image_size))
    ).size() == (
        batch_size,
        args.in_channels * 8,
        image_size // 8,
        image_size // 8,
    ), "Encoder block is not working correctly".capitalize()
