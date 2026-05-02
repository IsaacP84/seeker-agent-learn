#!/usr/bin/env python3
import os
import sys
import tempfile
import random

# Ensure the scripts root (apps/assets/scripts) is on sys.path so 'learning' package imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from learning.agent import Agent, _load_checkpoint_file, RUNS_DIR


def run_test(custom_runs_dir=None):
    runs_dir = custom_runs_dir or tempfile.mkdtemp(prefix='runs_test_')
    os.makedirs(runs_dir, exist_ok=True)

    # Override module-level RUNS_DIR so Agent writes to our temp folder
    import learning.agent as agent_mod
    agent_mod.RUNS_DIR = runs_dir

    # Create agent and force small checkpointing interval for test
    a = Agent('seeker')
    a.full_checkpoint_interval = 1
    a.save_replay_buffer_on_full = True
    a.compress = 'zip'

    # Minimal environment size for testing
    num_states = 8
    num_actions = 4
    a.setup(num_states, num_actions, is_training=True)

    # Generate 2 short episodes to trigger rolling + full checkpoints
    for ep in range(2):
        for step in range(5):
            s = [random.random() for _ in range(num_states)]
            ns = [random.random() for _ in range(num_states)]
            action = random.randrange(num_actions)
            reward = random.random()
            done = (step == 4)
            a.observe(s, action, reward, ns, done)

    files = os.listdir(runs_dir)
    print('Runs dir:', runs_dir)
    print('Files:', files)

    # Locate a full checkpoint (ep filename)
    full_ckpts = [f for f in files if f.startswith(a.hyperparameter_set + '_ep') and (f.endswith('.pt') or f.endswith('.zip'))]
    if not full_ckpts:
        print('ERROR: No full checkpoint found')
        return 2
    full_path = os.path.join(runs_dir, sorted(full_ckpts)[-1])

    try:
        ckpt = _load_checkpoint_file(full_path)
    except Exception as e:
        print('ERROR: Failed to load full checkpoint:', e)
        return 3

    if 'policy_dqn' not in ckpt:
        print('ERROR: Full checkpoint missing policy_dqn')
        return 4
    if a.save_replay_buffer_on_full and 'replay_buffer' not in ckpt:
        print('ERROR: Full checkpoint missing replay_buffer despite config')
        return 5

    # Check rolling checkpoint exists
    rolling_zip = os.path.join(runs_dir, a.hyperparameter_set + '.pt.zip')
    rolling_pt = os.path.join(runs_dir, a.hyperparameter_set + '.pt')
    if os.path.exists(rolling_zip):
        rolling_path = rolling_zip
    elif os.path.exists(rolling_pt):
        rolling_path = rolling_pt
    else:
        print('ERROR: No rolling checkpoint found')
        return 6

    try:
        rck = _load_checkpoint_file(rolling_path)
    except Exception as e:
        print('ERROR: Failed to load rolling checkpoint:', e)
        return 7

    if 'policy_dqn' not in rck:
        print('ERROR: Rolling checkpoint missing policy_dqn')
        return 8

    print('Checkpoint tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(run_test())
