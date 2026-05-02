#!/usr/bin/env python3
"""Checkpoint round-trip test for the PPO Agent."""
import os
import sys
import tempfile
import random

# Ensure the scripts root is on sys.path so 'learning' package imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

import torch
from learning.agent import Agent, RUNS_DIR


def run_test(custom_runs_dir=None):
    runs_dir = custom_runs_dir or tempfile.mkdtemp(prefix='runs_test_')
    os.makedirs(runs_dir, exist_ok=True)

    # Override module-level RUNS_DIR so Agent writes to our temp folder
    import learning.agent as agent_mod
    agent_mod.RUNS_DIR = runs_dir

    num_states  = 8
    num_actions = 4

    a = Agent('seeker')
    a.setup(num_states, num_actions, is_training=True)

    # Generate 2 short episodes — each episode flushes any partial buffer with last_value=0
    for ep in range(2):
        for step in range(5):
            s      = [random.random() for _ in range(num_states)]
            ns     = [random.random() for _ in range(num_states)]
            action = a.step(s)
            reward = random.random()
            done   = (step == 4)
            a.observe(s, action, reward, ns, done)

    files = os.listdir(runs_dir)
    print('Runs dir:', runs_dir)
    print('Files:', files)

    # Rolling checkpoint (written every episode)
    rolling_pt = os.path.join(runs_dir, a.hyperparameter_set + '.pt')
    if not os.path.exists(rolling_pt):
        print('ERROR: No rolling checkpoint found')
        return 1

    ckpt = torch.load(rolling_pt, map_location='cpu')

    for key in ('actor', 'critic', 'optimizer', 'rewards_per_episode'):
        if key not in ckpt:
            print(f'ERROR: Rolling checkpoint missing key: {key}')
            return 2

    if 'policy_dqn' in ckpt:
        print('ERROR: Rolling checkpoint contains DQN key "policy_dqn" — wrong checkpoint type')
        return 3

    # Verify actor weights round-trip correctly
    from learning.ppo_network import Actor, Critic
    actor2  = Actor( num_states, num_actions, a.hidden_dims)
    critic2 = Critic(num_states,              a.hidden_dims)
    try:
        actor2.load_state_dict(ckpt['actor'])
        critic2.load_state_dict(ckpt['critic'])
    except Exception as e:
        print(f'ERROR: Failed to restore actor/critic from checkpoint: {e}')
        return 4

    print('Checkpoint tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(run_test())
