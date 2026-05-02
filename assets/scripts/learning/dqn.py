import torch
from torch import nn


class DQN(nn.Module):
    """Flexible DQN MLP.

    Parameters
    - state_dim: int input size
    - action_dim: int output size
    - hidden_dims: int or list[int] specifying hidden layer sizes. If an int
      is provided it's treated as a single hidden layer. If None, defaults to 256.
    - hidden_dim: backward-compatible alias for hidden_dims (keyword only).
    - activation: name or nn.Module class for hidden activations (default 'relu').

    Examples
    - `DQN(s_dim, a_dim, hidden_dims=256)` -> one hidden layer
    - `DQN(s_dim, a_dim, hidden_dims=[256,128])` -> two hidden layers
    """

    def __init__(self, state_dim, action_dim, hidden_dims=None, hidden_dim=None, activation='relu'):
        super(DQN, self).__init__()

        # Backwards compatibility: allow `hidden_dim` keyword used previously
        if hidden_dims is None:
            hidden_dims = hidden_dim
        if hidden_dims is None:
            hidden_dims = 256

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Resolve activation layer
        if isinstance(activation, str):
            act = activation.lower()
            if act == 'relu':
                act_layer = nn.ReLU
            elif act == 'leakyrelu' or act == 'leaky_relu':
                act_layer = nn.LeakyReLU
            elif act == 'gelu':
                act_layer = nn.GELU
            elif act == 'tanh':
                act_layer = nn.Tanh
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            act_layer = activation
        else:
            raise ValueError("activation must be a string name or an nn.Module class")

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_layer())
            in_dim = h

        # Body contains hidden layers (may be empty if hidden_dims == [])
        self.body = nn.Sequential(*layers) if layers else nn.Identity()

        # Final head to produce action values
        self.head = nn.Linear(in_dim, action_dim)

    def forward(self, x):
        x = self.body(x)
        return self.head(x)