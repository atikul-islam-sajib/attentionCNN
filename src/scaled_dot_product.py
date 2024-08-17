import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, channels: int
):
    if (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        result = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(channels)

        assert result.size() == (
            query.size(0),
            query.size(1),
            query.size(2),
            query.size(2),
        ), "result size is not correct".capitalize()

        result = torch.softmax(result, dim=-1)

        attention = torch.matmul(result, value)

        assert attention.size() == (
            query.size(0),
            query.size(1),
            query.size(2),
            value.size(3),
        ), "attention size is not correct".capitalize()

        return attention


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaled dot product attention for Multihead attention layer".capitalize()
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Number of channels in the input tensor".capitalize(),
    )

    batch_size = config()["dataloader"]["batch_size"]
    image_size = config()["dataloader"]["image_size"]
    nheads = config()["attentionCNN"]["nheads"]

    scaled = scaled_dot_product_attention(
        query=torch.randn(
            batch_size, nheads, image_size // nheads, image_size * image_size
        ),
        key=torch.randn(
            batch_size, nheads, image_size // nheads, image_size * image_size
        ),
        value=torch.randn(
            batch_size, nheads, image_size // nheads, image_size * image_size
        ),
        channels=image_size,
    )

    assert scaled.size() == (
        batch_size,
        nheads,
        image_size // nheads,
        image_size * image_size,
    ), "scaled dot product attention output size is incorrect".capitalize()
