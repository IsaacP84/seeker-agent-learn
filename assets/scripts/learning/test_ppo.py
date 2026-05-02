"""Unit tests for PPO components (ppo_network, trajectory_buffer)."""
import math
import pytest
import torch

from learning.ppo_network import Actor, Critic
from learning.trajectory_buffer import RolloutBuffer

STATE_DIM  = 199
ACTION_DIM = 6
BATCH_SIZE = 16
N_STEPS    = 64
DEVICE     = "cpu"


# ---------------------------------------------------------------------------
# Actor tests
# ---------------------------------------------------------------------------

def test_actor_forward_shape():
    actor  = Actor(STATE_DIM, ACTION_DIM)
    obs    = torch.randn(BATCH_SIZE, STATE_DIM)
    logits = actor(obs)
    assert logits.shape == (BATCH_SIZE, ACTION_DIM)


def test_actor_get_action_values():
    actor = Actor(STATE_DIM, ACTION_DIM)
    obs   = torch.randn(1, STATE_DIM)
    action, log_prob, entropy = actor.get_action(obs)
    assert 0 <= int(action.item()) <= ACTION_DIM - 1
    assert math.isfinite(log_prob.item())
    assert entropy.item() > 0


def test_actor_evaluate_action_shape():
    actor   = Actor(STATE_DIM, ACTION_DIM)
    obs     = torch.randn(BATCH_SIZE, STATE_DIM)
    actions = torch.randint(0, ACTION_DIM, (BATCH_SIZE,))
    log_prob, entropy = actor.evaluate_action(obs, actions)
    assert log_prob.shape == (BATCH_SIZE,)
    assert entropy.shape  == (BATCH_SIZE,)


# ---------------------------------------------------------------------------
# Critic tests
# ---------------------------------------------------------------------------

def test_critic_forward_shape():
    critic = Critic(STATE_DIM)
    obs    = torch.randn(BATCH_SIZE, STATE_DIM)
    value  = critic(obs)
    assert value.shape == (BATCH_SIZE, 1)


# ---------------------------------------------------------------------------
# RolloutBuffer tests
# ---------------------------------------------------------------------------

def _fill_buffer(n_steps=N_STEPS):
    buf = RolloutBuffer(n_steps=n_steps, device=DEVICE)
    for _ in range(n_steps):
        buf.add(
            obs      = [0.0] * STATE_DIM,
            action   = 0,
            reward   = 1.0,
            done     = False,
            log_prob = -0.5,
            value    = 0.8,
        )
    return buf


def test_buffer_is_full():
    buf = _fill_buffer()
    assert buf.is_full()


def test_buffer_compute_advantages_shapes():
    buf = _fill_buffer()
    adv, ret = buf.compute_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == (N_STEPS,)
    assert ret.shape == (N_STEPS,)


def test_buffer_advantages_finite():
    buf = _fill_buffer()
    adv, ret = buf.compute_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(ret).all()


def test_buffer_get_batches():
    buf = _fill_buffer()
    buf.prepare(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    batches = list(buf.get_batches(mini_batch_size=16))
    assert len(batches) >= 1
    for batch in batches:
        assert set(batch.keys()) == {'obs', 'actions', 'log_probs', 'values', 'advantages', 'returns'}
        assert batch['obs'].shape[1] == STATE_DIM


def test_buffer_clear():
    buf = _fill_buffer()
    buf.clear()
    assert buf._size == 0
    assert not buf.is_full()


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_actor_checkpoint_roundtrip(tmp_path):
    actor = Actor(STATE_DIM, ACTION_DIM)
    obs   = torch.randn(1, STATE_DIM)

    with torch.no_grad():
        logits_before = actor(obs).clone()

    path = tmp_path / "actor.pt"
    torch.save(actor.state_dict(), path)

    actor2 = Actor(STATE_DIM, ACTION_DIM)
    actor2.load_state_dict(torch.load(path, map_location="cpu"))

    with torch.no_grad():
        logits_after = actor2(obs)

    assert torch.allclose(logits_before, logits_after)


def test_critic_checkpoint_roundtrip(tmp_path):
    critic = Critic(STATE_DIM)
    obs    = torch.randn(1, STATE_DIM)

    with torch.no_grad():
        val_before = critic(obs).clone()

    path = tmp_path / "critic.pt"
    torch.save(critic.state_dict(), path)

    critic2 = Critic(STATE_DIM)
    critic2.load_state_dict(torch.load(path, map_location="cpu"))

    with torch.no_grad():
        val_after = critic2(obs)

    assert torch.allclose(val_before, val_after)
