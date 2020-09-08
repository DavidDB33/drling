import drling
import gym
import os.path
import numpy as np

DIRNAME = os.path.join("tests", "res")

def _get_Box(n):
    return gym.spaces.Box(low=np.zeros(n, dtype=np.float32), high=np.ones(n, dtype=np.float32), dtype=np.float32)

def test__initialize():
    obs_space = _get_Box(5)
    act_space = gym.spaces.MultiDiscrete([3,3])
    config = drling.get_config(path=os.path.join(DIRNAME, "config.yaml"))
    model = drling.get_model("DQNv1", obs_space, act_space, config)
    agent = drling.get_agent(label="Agentv1", model=model, memory="Memoryv1")
