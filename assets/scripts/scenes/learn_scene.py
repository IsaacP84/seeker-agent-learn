import json
import os
from learning.agent_ppo import Agent
from Magic import ASSETS_FOLDER
from Magic import NavigationEnv as env


# Set by C++ (LearnScene/TestScene::onActivate) after the env is bound each activation.
scene_env  = None
scene_envs = []  # all agent envs; set alongside scene_env
_bound_env_id = None  # id() of the env we last bound; resets when scene_env changes

def _try_bind_env():
    global _bound_env_id
    if scene_env is not None and id(scene_env) != _bound_env_id:
        scene_agent.bind_env(scene_env)
        # Sync restored env_data to all agent envs so curriculum is consistent.
        if scene_envs:
            env_data = scene_env.get_env_data()
            for other_env in scene_envs[1:]:
                other_env.set_env_data(env_data)
        _bound_env_id = id(scene_env)

_config_path = os.path.join(ASSETS_FOLDER, "learning", "config.json")
with open(_config_path) as _f:
    _config = json.load(_f)

IS_TRAINING = bool(_config.get("is_training", True))
LOAD_MODEL  = _config.get("load_model") or None
if LOAD_MODEL:
    LOAD_MODEL = os.path.join(ASSETS_FOLDER, LOAD_MODEL)
NUM_AGENTS  = int(_config.get("num_agents", 1))
CHECKPOINT_CONFIG = _config.get("checkpointing", {})

scene_agent = Agent('seeker')
scene_agent.setup(
    env.num_states,
    env.num_actions,
    IS_TRAINING,
    LOAD_MODEL,
    num_agents=NUM_AGENTS,
    checkpoint_config=CHECKPOINT_CONFIG)
def step(obs, agent_id=0):
    _try_bind_env()
    return scene_agent.step(obs, agent_id=agent_id)

# Called from C++ once the next frame's transition is available.
def feedback(prev_obs, action, reward, done, next_obs, agent_id=0):
    _try_bind_env()
    scene_agent.observe(prev_obs, action, reward, next_obs, done, agent_id=agent_id)

# Called from C++ when ALL agents have finished their episode.
# Flushes any remaining rollout buffer data before the synchronized reset.
def round_complete():
    scene_agent.on_round_complete()