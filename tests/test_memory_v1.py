import drling
import gym
import os.path
import numpy as np
# import tensorflow as tf

DIRNAME = os.path.join("tests", "res")

def _get_env(config):
    return gym.make(config['environment']['name'])

def _get_memory():
    config = drling.get_config(path=os.path.join(DIRNAME, "config.yaml"))
    env = _get_env(config)
    obs_space = env.observation_space
    act_space = env.action_space
    memory = drling.get_memory("Memoryv1", config=config)
    model = drling.get_model("DQNv2", obs_space, act_space, config=config)
    agent = drling.get_agent(label="Agentv2", model=model, memory=memory)
    return memory, agent, env

def test_fill_memory():
    config = drling.get_config(path=os.path.join(DIRNAME, "config.yaml"))
    env = _get_env(config)
    obs_space = env.observation_space
    act_space = env.action_space
    memory1 = drling.get_memory("Memoryv1", config=config)
    model1 = drling.get_model("DQNv1", obs_space, act_space, config=config)
    agent1 = drling.get_agent(label="Agentv1", model=model1, memory=memory1)
    memory1.fill_memory(env=env, agent=agent1)
    memory2 = drling.get_memory("Memoryv1", config=config)
    model2 = drling.get_model("DQNv2", obs_space, act_space, config=config)
    agent2 = drling.get_agent(label="Agentv2", model=model2, memory=memory2)
    memory2.fill_memory(env=env, agent=agent2)

if __name__ == "__main__":
    memory, agent, env = _get_memory()
    memory.fill_memory(env=env, agent=agent)
