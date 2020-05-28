import sys
import argparse
import gym
import statistics as stt
import os.path
from tqdm import tqdm as Tqdm
import logging
import tensorflow as tf
import numpy as np
from drling import utils
from drling.utils import get_config, get_model, get_agent, get_memory, get_monitor, analyze_env

logging.basicConfig(format='%(asctime)s:%(name)s: %(levelname).1s %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S',
                    filename='training.log',
                    filemode='w')

formatter = logging.Formatter('%(asctime)s:%(name)s: %(levelname).1s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
# logger.info("########################### TRAINING #############################")

ENV_NAME = os.getenv("ENV_NAME", "CartPole-v1")
env = gym.make(ENV_NAME)
analyze_env(env)
env.close()

def run(env, agent, render=True):
    states, actions, rewards = list(), list(), list()
    obs = env.reset()
    env.render()
    belief = agent.guess(obs)
    done = False
    # tqdm = Tqdm(total = len(env.consumption_norm))
    while not done:
        # tqdm.update()
        belief = agent.guess(obs, belief)
        belief_nn = np.expand_dims(belief, axis=0)
        action = int(agent.act(agent(belief_nn)).numpy().squeeze())
        obs, reward, done, info = env.step(action)
        env.render()
        states.append(obs.tolist()), actions.append(action), rewards.append(reward)
    # tqdm.close()
    return states, actions, rewards

def run_agent():
    env = gym.make(ENV_NAME)
    config = get_config(path=os.path.join("config", ENV_NAME.lower()+".yaml"))
    observation_space, action_space = env.observation_space, env.action_space
    model = get_model("DQNv1", observation_space, action_space, config)
    model_path = os.path.join(config.resources.training.results, "early.h5")
    try:
        model.load_weights(model_path)
    except:
        pass
    memory = get_memory("Memoryv1", config)
    agent = get_agent("Agentv1", model, memory, config)
    monitor = get_monitor("MonitorV1", observation_space=observation_space, action_space=action_space, config=config)
    states, actions, rewards = run(env, agent)
    results = (states, actions, rewards)
    monitor.running(results)
    env.close()

def train(env, agent):
    obs_list, action_list, reward_list, loss_list = list(), list(), list(), list()
    belief = agent.guess()
    next_obs = env.reset()
    done = False # Reset
    for _ in range(agent.batch_size):
        if done:
            next_obs = env.reset()
        obs = next_obs
        belief = agent.guess(next_obs, belief)
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        agent.add_experience(belief, action, reward, next_obs, done)
    belief = agent.guess()
    next_obs = env.reset()
    done = False # Reset
    # tqdm = Tqdm(total = len(env.consumption_norm))
    while not done:
        # tqdm.update()
        obs = next_obs
        belief = agent.guess(next_obs, belief)
        belief_nn = np.expand_dims(belief, axis=0)
        if agent.explore(increment=True) > np.random.rand(): # performs an exploration step
            action = agent.sample()
        else:
            action = int(agent.act(agent(belief_nn)).numpy().squeeze())
        next_obs, reward, done, _ = env.step(action)
        agent.add_experience(belief, action, reward, next_obs, done)
        loss = agent.train_step()
        obs_list.append(next_obs.tolist()), action_list.append(action), reward_list.append(reward), loss_list.append(loss)
    # tqdm.close()
    return obs_list, action_list, reward_list, loss_list

def train_agent():
    config = get_config(path=os.path.join("config", ENV_NAME.lower()+".yaml"))
    suffix = ".w{:02}.h5".format(config.agent.history_window) if hasattr(config.agent, "history_window") else ".h5"
    os.makedirs(config.resources.training.results, exist_ok=True)
    weights_path = os.path.join(config.resources.training.results, "weigths" + suffix)
    early_path = os.path.join(config.resources.training.results, "early" + suffix)
    trainer, developer, tester = gym.make(ENV_NAME), gym.make(ENV_NAME), gym.make(ENV_NAME)
    observation_space, action_space = trainer.observation_space, trainer.action_space
    memory = get_memory("Memoryv1", config=config)
    model = get_model("DQNv1", observation_space=observation_space, action_space=action_space, config=config)
    try:
        model.load_weights(weights_path)
    except:
        pass
    agent = get_agent("Agentv1", model=model, memory=memory, config=config)
    monitor = get_monitor("MonitorV1", observation_space=observation_space, action_space=action_space, config=config)

    best_reward = None
    while not monitor.done:
        obs_list, action_list, reward_list, loss_list = train(trainer, agent)
        training_results = obs_list, action_list, reward_list, loss_list
        model.save_weights(weights_path)
        obs_list, action_list, reward_list = run(developer, agent)
        developing_results = (obs_list, action_list, reward_list)
        monitor.training(training_results, developing_results)
        if monitor.improved:
            model.save_weights(early_path)
        monitor.debug(env, agent)
        monitor.plot(developing_results)
    env.close()

if __name__ == "__main__":
    args, parser = utils.parse_arguments()
    if args.exec == "run":
        run_agent()
    elif args.exec == "train":
        train_agent()
    else:
        parser.print_help()
