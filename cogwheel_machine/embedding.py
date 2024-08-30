"""Embedding networks."""
import torch
from torch import nn


class FullyConnectedEmbeddingNetwork(nn.Module):
    """Embedding network with hidden layers of variable size."""
    def __init__(self, input_size, layer_sizes):
        """
        Parameters
        ----------
        input_size: int
            The size of the input features.

        layer_sizes: list of int
            Each element is the size of the corresponding hidden layer.
        """
        super().__init__()

        # Create a list of fully connected layers
        layers = []
        in_size = input_size
        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(in_size, size))
            # Add ReLU only after hidden layers:
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            in_size = size

        # Combine the layers into a sequential model
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor: Output of shape (batch_size, final_layer_size).
        """
        return self.fc_layers(x)


class BlockMatrixEmbeddingNetwork(nn.Module):
    """Embedding network with block matrix structure."""
    def __init__(self, input_size, layer_sizes, unchanged_size):
        """
        Parameters
        ----------
        input_size: int
            The size of the input features.
            
        layer_sizes: list of int
            Each element is the size of the corresponding hidden layer.

        unchanged_size: int
            The size of the part of the input that should remain
            unaffected.
        """
        super().__init__()

        self.unchanged_size = unchanged_size

        # Create a list of fully connected layers for the processed part
        layers = []
        in_size = input_size
        for size in layer_sizes:
            layers.append(nn.Linear(in_size, size))
            in_size = size + unchanged_size

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor:
            Of shape (batch_size, final_layer_size + unchanged_size).
        """
        # Split the input into two parts
        x_processed = x
        x_unchanged = x[:, -self.unchanged_size:]
        relu = nn.ReLU()

        # Process the upper part with the influence of the unchanged part
        for i, layer in enumerate(self.fc_layers):
            x_processed = layer(x_processed)
            # Add ReLU only after hidden layers:
            if i < len(self.fc_layers) - 1:
                x_processed = relu(x_processed)
            # Concatenate unchanged part to the processed part
            x_processed = torch.cat((x_processed, x_unchanged), dim=1)

        # Combine the processed and unchanged parts
        return x_processed
