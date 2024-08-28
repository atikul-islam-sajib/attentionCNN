import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int = 1024, out_channels: int = 512, batchnorm: bool = True
    ):
        super(DecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batchnorm = batchnorm

        self.kernel_size = 4
        self.stride_size = 2
        self.padding_size = 1

        self.layers = []

        self.decoder_block = self.layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
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
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )

        if self.use_batchnorm:
            self.layers.append(nn.BatchNorm2d(num_features=self.out_channels))

        self.decoder_block = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.decoder_block(x)

        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decoder Block for attentionCNN".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1024,
        help="Number of input channels for the decoder block".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=512,
        help="Number of output channels for the decoder block".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels

    batch_size = config()["dataloader"]["batch_size"]
    image_size = config()["dataloader"]["image_size"] // 8

    layers = []

    for _ in range(2):
        layers.append(DecoderBlock(in_channels=in_channels, out_channels=out_channels))

        in_channels = out_channels
        out_channels //= 2

    model = nn.Sequential(*layers)

    assert model(
        torch.randn(batch_size, args.in_channels, image_size, image_size)
    ).size() == (
        batch_size,
        args.in_channels // 4,
        image_size * 4,
        image_size * 4,
    ), "Decoder block output size is incorrect".capitalize()
