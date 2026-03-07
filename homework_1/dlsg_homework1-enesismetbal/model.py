"""
Model definition module.
Students must complete the MLP architectures below.
"""

import torch
import torch.nn as nn
from config import MODEL_CONFIG


class PhysicsMLP(nn.Module):
    """
    A standard MLP for regression (no regularization).

    Architecture (start with this then try bigger/smaller models):
        Input(4) → Linear(8) → ReLU → Linear(6) → ReLU → Linear(4) → ReLU → Linear(1) 
    """

    def __init__(self, input_dim=None, hidden_dims=None, output_dim=None):
        super(PhysicsMLP, self).__init__()

        if input_dim is None:
            input_dim = MODEL_CONFIG["input_dim"]
        if hidden_dims is None:
            hidden_dims = MODEL_CONFIG["hidden_dims"]
        if output_dim is None:
            output_dim = MODEL_CONFIG["output_dim"]

        layers = []
        curr_dim = input_dim

        for hidden in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden))
            layers.append(nn.ReLU())
            curr_dim = hidden

        layers.append(nn.Linear(curr_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.network(x)


class PhysicsMLPWithDropout(nn.Module):
    """
    MLP with Dropout regularization.

    Architecture (same architecture as PhysicsMLP, but with dropout layers):
    """

    def __init__(self, input_dim=None, hidden_dims=None, output_dim=None,
                 dropout_rate=None):
        super(PhysicsMLPWithDropout, self).__init__()

        if input_dim is None:
            input_dim = MODEL_CONFIG["input_dim"]
        if hidden_dims is None:
            hidden_dims = MODEL_CONFIG["hidden_dims"]
        if output_dim is None:
            output_dim = MODEL_CONFIG["output_dim"]
        if dropout_rate is None:
            dropout_rate = MODEL_CONFIG["dropout_rate"]

        layers = []
        curr_dim = input_dim

        for hidden in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = hidden

        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.network(x)


def create_model(model_type="standard", **kwargs):
    """
    Factory function to create a model by name.

    Args:
        model_type: One of "standard" or "dropout"
        **kwargs: Additional keyword arguments passed to the model constructor

    Returns:
        An instance of the requested model
    """
    if model_type == "standard":
        return PhysicsMLP(**kwargs)
    elif model_type == "dropout":
        return PhysicsMLPWithDropout(**kwargs)
    else:
        raise ValueError(f"Invalid model_type '{model_type}'. Expected 'standard' or 'dropout'.")
    pass
