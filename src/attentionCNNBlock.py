import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from multihead_attention import MultiHeadAttentionLayer
from feedforward_network import FeedForwardNeuralNetwork


class attentionCNNBlock(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        nheads: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ):
        super(attentionCNNBlock, self).__init__()

        self.channels = channels
        self.nheads = nheads
        self.dropout = dropout
        self.activation = activation
        self.bias = bias

        self.multihead_attention = MultiHeadAttentionLayer(
            channels=self.channels,
            nheads=self.nheads,
            bias=self.bias,
        )

        self.feedforward_network = FeedForwardNeuralNetwork(
            channels=self.channels,
            dropout=self.dropout,
            activation=self.activation,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            residual = x

            x = self.multihead_attention(x=x)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)
            x = nn.BatchNorm2d(num_features=self.channels)(x)

            residual = x

            x = self.feedforward_network(x=x)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)
            x = nn.BatchNorm2d(num_features=self.channels)(x)

            return x

        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="attentionCNNBlock for the attentionCNN".title()
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
        help="Number of heads in the multihead attention".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["attentionCNN"]["dropout"],
        help="Dropout rate for the feedforward network".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=config()["attentionCNN"]["activation"],
        help="Activation function for the feedforward network".capitalize(),
    )
    args = parser.parse_args()

    batch_size = config()["dataloader"]["batch_size"]
    image_size = config()["dataloader"]["image_size"]

    attention_cnn = attentionCNNBlock(
        channels=args.channels,
        nheads=args.nheads,
        dropout=args.dropout,
        activation=args.activation,
        bias=True,
    )

    assert attention_cnn(
        torch.randn(batch_size, args.channels, image_size, image_size)
    ).size() == (batch_size, args.channels, image_size, image_size)
