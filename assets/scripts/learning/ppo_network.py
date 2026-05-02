import torch
from torch import nn
from torch.distributions import Categorical


def _build_mlp(in_dim, hidden_dims, activation='relu'):
    """Build a sequential MLP body (hidden layers only, no output head)."""
    if isinstance(activation, str):
        act = activation.lower()
        if act == 'relu':
            act_layer = nn.ReLU
        elif act in ('leakyrelu', 'leaky_relu'):
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
        raise ValueError("activation must be a string name or an nn.Module subclass")

    layers = []
    dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(dim, h))
        layers.append(act_layer())
        dim = h
    body = nn.Sequential(*layers) if layers else nn.Identity()
    return body, dim


class Actor(nn.Module):
    """Policy network: obs → action probabilities (Categorical distribution).

    Parameters
    ----------
    state_dim : int
        Observation vector size (e.g. 200).
    action_dim : int
        Number of discrete actions (e.g. 6).
    hidden_dims : list[int]
        Hidden layer sizes, e.g. [256, 256, 128].
    activation : str
        Hidden activation name ('relu', 'leakyrelu', 'gelu', 'tanh').
    """

    def __init__(self, state_dim, action_dim, hidden_dims=None, activation='relu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.body, out_dim = _build_mlp(state_dim, hidden_dims, activation)
        self.head = nn.Linear(out_dim, action_dim)

    def forward(self, x):
        """Return raw logits (unnormalized). Shape: (batch, action_dim)."""
        return self.head(self.body(x))

    def get_action(self, obs):
        """Sample an action from the policy.

        Parameters
        ----------
        obs : torch.Tensor  shape (state_dim,) or (batch, state_dim)

        Returns
        -------
        action     : int (scalar) or LongTensor (batch,)
        log_prob   : float Tensor
        entropy    : float Tensor
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_action(self, obs, action):
        """Evaluate log-prob and entropy of a stored action.

        Used during the PPO update pass.

        Parameters
        ----------
        obs    : Tensor (batch, state_dim)
        action : LongTensor (batch,)

        Returns
        -------
        log_prob : Tensor (batch,)
        entropy  : Tensor (batch,)
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class Critic(nn.Module):
    """Value network: obs → scalar state-value estimate V(s).

    Parameters
    ----------
    state_dim   : int
    hidden_dims : list[int]
    activation  : str
    """

    def __init__(self, state_dim, hidden_dims=None, activation='relu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.body, out_dim = _build_mlp(state_dim, hidden_dims, activation)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        """Return value estimate. Shape: (batch, 1)."""
        return self.head(self.body(x))
