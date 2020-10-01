import drling
import gym
import os.path
import numpy as np
import tensorflow as tf

DIRNAME = os.path.join("tests", "res")

def _get_Box(n):
    return gym.spaces.Box(low=np.zeros(n, dtype=np.float32), high=np.ones(n, dtype=np.float32), dtype=np.float32)

def _get_spaces(o, a):
    obs_space = _get_Box(o)
    act_space = gym.spaces.MultiDiscrete(a)
    return obs_space, act_space

def _get_Agent2(spaces):
    obs_space, act_space = spaces
    config = drling.get_config(path=os.path.join(DIRNAME, "config.yaml"))
    model = drling.get_model("DQNv2", obs_space, act_space, config=config)
    agent = drling.get_agent(label="Agentv2", model=model, memory="Memoryv1")
    return agent

def test__ravel_action():
    spaces = _get_spaces(5, (3, 3))
    agent = _get_Agent2(spaces)
    assert agent._ravel_action([0,0]) == 0
    assert agent._ravel_action([0,1]) == 1
    assert agent._ravel_action([0,2]) == 2
    assert agent._ravel_action([1,0]) == 3
    assert agent._ravel_action([1,1]) == 4
    assert agent._ravel_action([2,1]) == 7
    assert agent._ravel_action([2,2]) == 8
