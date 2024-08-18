import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn

sys.path.append("./src/")

from encoder_block import EncoderBlock
from decoder_block import DecoderBlock
from attentionCNNBlock import attentionCNNBlock


class attentionCNN(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 128,
        nheads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 8,
        activation: str = "relu",
        bias: bool = True,
    ):
        super(attentionCNN, self).__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.nheads = nheads
        self.dropout = dropout
        self.num_layers = num_layers
        self.activation = activation
        self.bias = bias

        self.kernel_size = 3
        self.stride_size = 1
        self.padding_size = 1

        self.input_block = nn.Conv2d(
            in_channels=self.image_channels,
            out_channels=self.image_size,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
            bias=self.bias,
        )
        
        self.encoder1 = EncoderBlock(in_channels=self.image_size, out_channels=self.image_size*2, batch_norm=True)
        self.encoder2 = EncoderBlock(in_channels=self.image_size*2, out_channels=self.image_size*4, batch_norm=True)
        self.encoder3 = EncoderBlock(in_channels=self.image_size*4, out_channels=self.image_size*8, batch_norm=False)
        
        self.decoder1 = DecoderBlock(in_channels=self.image_size*8, out_channels=self.image_size*4, batchnorm=True)
        self.decoder2 = DecoderBlock(in_channels=self.image_size*8, out_channels=self.image_size*2, batchnorm=True)
        self.decoder3 = DecoderBlock(in_channels=self.image_size*4, out_channels=self.image_size, batchnorm=True)
        
         

        self.attention_cnn_block = nn.Sequential(
            *[
                attentionCNNBlock(
                    channels=self.image_size,
                    nheads=self.nheads,
                    dropout=self.dropout,
                    activation=self.activation,
                    bias=self.bias,
                )
                for _ in tqdm(range(self.num_layers))
            ]
        )

        self.output_block = nn.Conv2d(
            in_channels=self.image_size,
            out_channels=self.image_channels,
            kernel_size=self.kernel_size,
            stride=self.stride_size,
            padding=self.padding_size,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.input_block(x)
            x = self.attention_cnn_block(x)
            
            encoder1 = self.encoder1(x)
            encoder2 = self.encoder2(encoder1)
            encoder3 = self.encoder3(encoder2)
            
            decoder1 = self.decoder1(encoder3)
            decoder1 = torch.cat((decoder1, encoder2), dim=1)
            
            decoder2 = self.decoder2(decoder1)
            decoder2 = torch.cat((decoder2, encoder1), dim = 1)
            
            decoder3 = self.decoder3(decoder2)
            
            output = self.output_block(decoder3)

            return torch.tanh(input=output)

        else:
            raise ValueError("Input must be a torch.Tensor")


if __name__ == "__main__":
    attention_cnn = attentionCNN(
        image_channels=3,
        image_size=128,
        nheads=8,
        dropout=0.1,
        num_layers=1,
        activation="relu",
        bias=True,
    )

    print(attention_cnn(torch.randn(16, 3, 128, 128)).size())
