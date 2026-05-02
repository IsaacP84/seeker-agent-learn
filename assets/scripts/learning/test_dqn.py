import torch
from dqn import DQN


def _test_variant(state_dim, action_dim, hidden_dims):
    model = DQN(state_dim, action_dim, hidden_dims)
    x = torch.randn(4, state_dim)
    y = model(x)
    assert y.shape == (4, action_dim), f"Unexpected output shape: {y.shape}"


def main():
    state_dim = 10
    action_dim = 3

    # single hidden int
    _test_variant(state_dim, action_dim, 256)

    # multiple layers
    _test_variant(state_dim, action_dim, [256, 128, 64])

    # no hidden layers (edge-case): provide empty list -> direct linear
    _test_variant(state_dim, action_dim, [])

    print("test_dqn: all checks passed")


if __name__ == '__main__':
    main()
