import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from multihead_attention import MultiHeadAttentionLayer
from feedforward_network import FeedForwardNeuralNetwork

class attentionCNNBlock(nn.Module):
    def __init__(self, channels: int = 128, nheads: int = 8, dropout: float = 0.1, activation: str = "relu", bias: bool = True,):
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
    attention_cnn = attentionCNNBlock(
        channels=128,
        nheads=8,
        dropout=0.1,
        activation="relu",
        bias=True
    )
    
    assert attention_cnn(torch.randn(16, 128, 128, 128)).size() == (16, 128, 128, 128)
        
        