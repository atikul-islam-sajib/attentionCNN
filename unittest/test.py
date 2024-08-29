import os
import sys
import torch
import unittest
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("./src/")

from helper import helpers
from dataloader import Loader
from utils import config, load
from patch_embedding import PatchEmbedding
from transformer import TransformerEncoder
from positional_encoding import PositionalEncoding
from feedforward_network import FeedForwardNetwork
from layer_normalization import LayerNormalization
from multihead_attention import MultiHeadAttention
from encoder_block import TransformerEncoderBlock
from scaled_dot_product import scaled_dot_product_attention


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.image_size = 224
        self.patch_size = 16
        self.image_channels = 3
        self.total_texts = 400
        self.batch_size = 40
        self.nheads = 8
        self.num_classes = 4
        self.num_layers = 6
        self.sequence_length = 200
        self.feedforward = 2048
        self.dimension = 512
        self.constant = 10000
        self.dropout = 0.1
        self.activation = "relu"

        self.query = torch.randn(self.batch_size, self.sequence_length, self.dimension)
        self.key = torch.randn(self.batch_size, self.sequence_length, self.dimension)
        self.value = torch.randn(self.batch_size, self.sequence_length, self.dimension)

        self.positional_encoding = PositionalEncoding(
            sequence_length=self.sequence_length,
            dimension=self.dimension,
            constant=self.constant,
        )

        self.attention = scaled_dot_product_attention(
            query=self.query.view(
                self.query.size(0),
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            ),
            key=self.key.view(
                self.key.size(0),
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            ),
            value=self.value.view(
                self.value.size(0),
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            ),
        )

        self.network = FeedForwardNetwork(
            in_features=self.dimension,
            out_features=self.feedforward,
            activation=self.activation,
        )

        self.layernorm = LayerNormalization(
            normalized_shape=self.dimension,
        )

        self.multihead_attention = MultiHeadAttention(
            dimension=self.dimension,
            nheads=self.nheads,
            dropout=self.dropout,
        )

        self.encoder_block = TransformerEncoderBlock(
            dimension=self.dimension,
            nheads=self.nheads,
            dim_feedforward=self.feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.transformerEncoder = TransformerEncoder(
            dimension=self.dimension,
            nheads=self.nheads,
            dim_feedforward=self.feedforward,
            dropout=self.dropout,
            activation=self.activation,
            num_encoder_layers=self.num_layers,
        )

        self.patch_embedding = PatchEmbedding(
            image_size=self.image_size,
            image_channels=self.image_channels,
            patch_size=self.patch_size,
        )

        self.loader = Loader(
            image_path="./data/raw/dataset.zip",
            image_size=128,
            image_channels=self.image_channels,
            batch_size=64,
            split_size=0.25,
        )

        self.model = ViT(
            image_size=self.image_size,
            image_channels=self.image_channels,
            patch_size=self.patch_size,
            labels=self.num_classes,
            num_encoder_layers=self.num_layers,
            nheads=self.nheads,
            dim_feedforward=self.feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.init = helpers(
            image_channels=config()["dataloader"]["channels"],
            image_size=config()["dataloader"]["image_size"],
            labels=len(config()["dataloader"]["labels"]),
            patch_size=config()["ViT"]["patch_size"],
            nheads=config()["ViT"]["nheads"],
            num_encoder_layers=config()["ViT"]["num_layers"],
            dropout=config()["ViT"]["dropout"],
            dim_feedforward=config()["ViT"]["dim_feedforward"],
            epsilon=config()["ViT"]["eps"],
            activation=config()["ViT"]["activation"],
            bias=True,
            lr=config()["trainer"]["lr"],
            beta1=config()["trainer"]["beta1"],
            beta2=config()["trainer"]["beta2"],
            momentum=config()["trainer"]["momentum"],
            adam=config()["trainer"]["adam"],
            SGD=config()["trainer"]["SGD"],
        )

    def test_positional_encoding(self):
        embedding_layer = nn.Embedding(
            num_embeddings=self.sequence_length, embedding_dim=self.dimension
        )

        embedding = embedding_layer(
            torch.randint(
                0, self.sequence_length, (self.total_texts, self.sequence_length)
            )
        )

        self.assertEqual(
            embedding.size(), (self.total_texts, self.sequence_length, self.dimension)
        )

        positional_encoding = self.positional_encoding(x=embedding)

        self.assertEqual(
            positional_encoding.size(), (1, self.sequence_length, self.dimension)
        )

        embeddings_with_positional = torch.add(embedding, positional_encoding)

        self.assertEqual(
            embeddings_with_positional.size(),
            (400, self.sequence_length, self.dimension),
        )

    def test_positional_encoding_with_dataloader(self):
        embedding_layer = nn.Embedding(
            num_embeddings=self.sequence_length, embedding_dim=self.dimension
        )
        embedding = embedding_layer(
            torch.randint(
                0, self.sequence_length, (self.total_texts, self.sequence_length)
            )
        )

        dataloader = DataLoader(
            dataset=list(embedding), batch_size=self.batch_size, shuffle=True
        )

        data = next(iter(dataloader))

        self.assertEqual(
            data.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

        positional_encoding = self.positional_encoding(x=data)

        embeddings_with_positional = torch.add(data, positional_encoding)

        self.assertEqual(
            embeddings_with_positional.size(),
            (self.batch_size, self.sequence_length, self.dimension),
        )

    def test_scaled_dot_product(self):
        self.assertEqual(
            self.attention.size(),
            (
                self.batch_size,
                self.nheads,
                self.sequence_length,
                self.dimension // self.nheads,
            ),
        )

    def test_feedforward_neural_network(self):
        embedding_layer = nn.Embedding(
            num_embeddings=self.sequence_length, embedding_dim=self.dimension
        )

        embedding = embedding_layer(
            torch.randint(
                0, self.sequence_length, (self.total_texts, self.sequence_length)
            )
        )

        position = self.positional_encoding(x=embedding)

        embedding_with_position = torch.add(embedding, position)

        dataloader = DataLoader(
            dataset=list(embedding_with_position),
            batch_size=self.batch_size,
            shuffle=True,
        )

        data = next(iter(dataloader))

        self.assertEqual(
            data.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

        result = self.network(x=data)

        self.assertEqual(
            result.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

    def test_layer_normalization(self):
        result = self.network(
            x=torch.randn(self.batch_size, self.sequence_length, self.dimension)
        )

        normalization = self.layernorm(x=result)

        self.assertEqual(
            normalization.size(),
            (self.batch_size, self.sequence_length, self.dimension),
        )

        self.assertIsInstance(self.layernorm, LayerNormalization)

    def test_multihead_attention_layer(self):
        x = torch.randn(self.batch_size, self.sequence_length, self.dimension)

        mask = None

        attention = self.multihead_attention(x=x, mask=mask)

        self.assertEqual(
            attention.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

        embedding_layer = nn.Embedding(self.sequence_length, self.dimension)

        embedding = embedding_layer(
            torch.randint(
                0, self.sequence_length, (self.batch_size, self.sequence_length)
            )
        )

        position = self.positional_encoding(x=embedding)

        embedding_with_position = torch.add(embedding, position)

        attention = self.multihead_attention(embedding_with_position)

        self.assertEqual(
            attention.size(), (self.batch_size, self.sequence_length, self.dimension)
        )

        self.assertIsInstance(self.multihead_attention, MultiHeadAttention)
        self.assertIsInstance(embedding_layer, nn.Embedding)
        self.assertIsInstance(self.positional_encoding, PositionalEncoding)

    def test_encoder_block(self):
        X = torch.randint(
            0, self.sequence_length, (self.total_texts, self.sequence_length)
        )
        y = torch.randint(0, 4, (self.total_texts,))

        dataloader = DataLoader(
            dataset=list(zip(X, y)), batch_size=self.batch_size, shuffle=True
        )

        data, label = next(iter(dataloader))

        embedding_layer = nn.Embedding(
            num_embeddings=self.sequence_length, embedding_dim=self.dimension
        )

        embedding = embedding_layer(data)

        position = self.positional_encoding(x=embedding)

        embedding_with_position = torch.add(position, embedding)

        encoder_result = self.encoder_block(x=embedding_with_position)

        self.assertEqual(
            data.size(),
            (self.batch_size, self.sequence_length),
        )
        self.assertEqual(
            label.size(),
            (self.batch_size,),
        )
        self.assertEqual(
            embedding.size(),
            (self.batch_size, self.sequence_length, self.dimension),
        )
        self.assertEqual(
            embedding_with_position.size(),
            (self.batch_size, self.sequence_length, self.dimension),
        )
        self.assertEqual(
            encoder_result.size(),
            (self.batch_size, self.sequence_length, self.dimension),
        )

    def test_transformerEncoder(self):
        self.assertEqual(
            self.transformerEncoder(
                x=torch.randn(self.batch_size, self.sequence_length, self.dimension)
            ).size(),
            (self.batch_size, self.sequence_length, self.dimension),
        )

        self.assertIsInstance(self.transformerEncoder, TransformerEncoder)

    def test_patch_embedding(self):
        num_of_patches = (self.image_size // self.patch_size) ** 2
        num_of_dimension = (self.patch_size**2) * self.image_channels

        image = torch.randn(
            self.batch_size, self.image_channels, self.image_size, self.image_size
        )

        self.assertEqual(
            self.patch_embedding(image).size(),
            (self.batch_size, num_of_patches + 1, num_of_dimension),
        )

    def test_dataloader(self):
        train_dataloader = load(
            filename=os.path.join(
                config()["path"]["PROCESSED_DATA_PATH"], "train_dataloader.pkl"
            )
        )

        train_data, _ = next(iter(train_dataloader))

        channels = train_data.size(1)
        batch_size = train_data.size(0)
        image_size = train_data.size(2)

        self.assertEqual(channels, self.image_channels)
        self.assertEqual(batch_size, 64)
        self.assertEqual(image_size, 128)

    def test_ViT_model(self):
        self.assertEqual(
            self.model(
                x=torch.randn(
                    self.batch_size,
                    self.image_channels,
                    self.image_size,
                    self.image_size,
                )
            ).size(),
            (self.batch_size, self.num_classes),
        )

        self.assertIsInstance(self.model, ViT)

    def test_helper_function(self):

        assert (
            self.init["train_dataloader"].__class__ == torch.utils.data.DataLoader
        ), "Dataloader is not a torch.utils.data.DataLoader".capitalize()
        assert (
            self.init["valid_dataloader"].__class__ == torch.utils.data.DataLoader
        ), "Dataloader is not a torch.utils.data.DataLoader".capitalize()

        assert self.init["model"].__class__ == ViT, "Model is not a ViT".capitalize()
        assert (
            self.init["optimizer"].__class__ == torch.optim.Adam
        ), "Optimizer is not a Adam".capitalize()
        assert (
            self.init["criterion"].__class__ == CategoricalLoss
        ), "Loss is not a CategoricalLoss".capitalize()

    def test_categorical_loss(self):
        reduction = "mean"

        loss = CategoricalLoss(reduction=reduction)

        batch_size = config()["dataloader"]["batch_size"]
        total_labels = len(config()["dataloader"]["labels"])

        actual = torch.randn((batch_size, total_labels))
        predicted = torch.randn((batch_size, total_labels))

        actual = torch.softmax(actual, dim=1)
        predicted = torch.softmax(predicted, dim=1)

        actual = torch.argmax(actual, dim=1).float()
        predicted = torch.argmax(predicted, dim=1).float()

        assert isinstance(
            loss(actual, predicted), torch.Tensor
        ), "Loss is not a torch.Tensor".capitalize()


if __name__ == "__main__":
    unittest.main()
