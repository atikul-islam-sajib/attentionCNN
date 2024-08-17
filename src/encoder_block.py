import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


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
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
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
    in_channels = 128
    out_channels = 256

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

    assert (model(torch.randn(16, 128, 128, 128))).size() == (
        16,
        in_channels,
        128 // 8,
        128 // 8,
    )
