"""Embedding networks."""
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
