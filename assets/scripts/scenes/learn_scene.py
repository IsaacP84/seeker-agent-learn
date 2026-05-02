import json
import os
import random
from learning.agent import Agent
from Magic import ASSETS_FOLDER
from Magic import NavigationEnv as env


# Set by C++ (LearnScene::onActivate) after the env is bound each activation.
scene_env = None
_env_bound = False

def _try_bind_env():
    global _env_bound
    if not _env_bound and scene_env is not None:
        scene_agent.bind_env(scene_env)
        _env_bound = True

_config_path = os.path.join(ASSETS_FOLDER, "learning", "config.json")
with open(_config_path) as _f:
    _config = json.load(_f)

IS_TRAINING = bool(_config.get("is_training", True))
LOAD_MODEL  = _config.get("load_model") or None
if LOAD_MODEL:
    LOAD_MODEL = os.path.join(ASSETS_FOLDER, LOAD_MODEL)

scene_agent = Agent('seeker')
scene_agent.setup(env.num_states, env.num_actions, IS_TRAINING, LOAD_MODEL)
def step(obs):
    _try_bind_env()
    return scene_agent.step(obs)

# Called from C++ once the next frame's transition is available.
def feedback(prev_obs, action, reward, done, next_obs):
    _try_bind_env()
    scene_agent.observe(prev_obs, action, reward, next_obs, done)