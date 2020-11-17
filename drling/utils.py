#!/usr/bin/env python
import argparse
import copy
import math
import os.path
import random
import sys
import time
import yaml
from collections import namedtuple
from collections.abc import Iterable
import gym
import numpy as np
from tqdm import autonotebook as tqdm
from .core import Agentv1, Agentv2, DQNv1, DQNv2, Memoryv1, Monitorv1

_config_file = None # Singleton
_config = None # Singleton
_parser = None # Singleton
_seed = None # Singleton

class GetAgentError(Exception):
    pass

def set_parser(parser):
    global _parser
    _parser = parser

def get_seed():
    global _seed
    if _seed is None:
        _seed = random.randint(0, 2**32-1)
    return _seed

def get_parser(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Drling framework")
    parser.add_argument("-w", "--windows_size", dest="history_window", type=int, default=None)
    parser.add_argument("-m", "--model-path", dest="model_path", type=str, default=None)
    parser.add_argument("-c", "--config-file-path", dest="config_filename", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=None)
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help="Train the agent")
    train_parser.add_argument("-n", "--n-epochs", type=int, default=None)
    train_parser.add_argument("-a", "--learning-rate", type=float, default=None, help="learning rate (α) of the ANN")
    train_parser.set_defaults(exec="train")
    run_parser = subparsers.add_parser('run', help="Run the agent")
    run_parser.set_defaults(exec="run")
    defaults = dict(
        exec="run",
        seed=None,
        learning_rate=None,
        n_epochs=None,
    )
    parser.set_defaults(**defaults)
    return parser

def update_config_with_args(config, args):
    if args.history_window is not None: config['agent']['history_window'] = args.history_window
    if args.model_path is not None: config['resources']['running']['model_path'] = args.model_path
    if args.learning_rate is not None: config['agent']['network']['learning_rate'] = args.learning_rate
    if args.seed is not None: config['seed'] = args.seed
    if args.n_epochs is not None: config['max_epochs'] = args.n_epochs
    return config

def get_config_from_file(path = None):
    if path is None:
        filename = os.path.join("config", "agent.config.yaml")
    else:
        filename = os.path.expanduser(path)
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    if 'seed' not in config or config['seed'] is None: config['seed'] = get_seed()
    return config

def get_config(path = None):
    """If path is specified, returns the config from path, else get a copy of the last config loaded as a singleton.

    Program parameters have priority over the file
    Args:
        path (str): Path to the configuration file. If path is None and a config was never loaded, use default of get_config_from_file()

    Returns:
        _config (dict): A dict with the file config updated by program params
    """
    global _config
    if path is None and _config is not None:
        return copy.deepcopy(_config)
    _config = get_config_from_file(path)
    return copy.deepcopy(_config)

def set_config(config):
    global _config
    _config = config
    
def get_agent(label, model, memory=None, config=None, verbose=False):
    """By default, load the default config"""
    Agent_class = {"Agentv1": Agentv1, "Agentv2": Agentv2}
    if memory is None:
        raise GetAgentError("Memory is None, call get_memory function to retrieve a memory, or insert a memory label")
    if config is None:
        config = get_config()
    if isinstance(memory, str):
        memory = get_memory(memory, config, verbose=verbose)
    try:
        agent = Agent_class[label](model=model, memory=memory, config=config)
    except KeyError as e:
        raise (LabelError("Label {} not found. Must be {}".format(e, list(Agent_class.keys()))
            ).with_traceback(sys.exc_info()[2]))
    return agent

def get_agent_from_config(env, config, verbose=False):
    model = get_model(label=config['agent']['network']['name'], observation_space=env.observation_space, action_space=env.action_space, config=config)
    agent = get_agent(label=config['agent']['name'], model=model, memory=config['agent']['memory']['name'], config=config, verbose=verbose)
    return agent

class LabelError(KeyError):
    pass

def get_model(label, observation_space, action_space, name="model", config=None):
    DQN_class = {"DQNv1": DQNv1, "DQNv2": DQNv2}
    try:
        model = DQN_class[label](observation_space, action_space, name=name, config=config)
    except KeyError as e:
        raise (LabelError("Label {} not found. Must be {}".format(e, list(DQN_class.keys()))
            ).with_traceback(sys.exc_info()[2]))
    return model

def get_memory(label, config, verbose=False):
    return Memoryv1(
        max_size=config['agent']['memory']['max_size'],
        min_size=config['agent']['memory']['min_size'],
        seed=config['seed'],
        verbose=verbose)

def get_monitor(label, agent, env_eval, config):
    return Monitorv1(agent, env_eval, config)

def get_monitor_from_config(agent, env_eval, config):
    return get_monitor(config['agent']['monitor']['name'], agent, env_eval=env_eval, config=config)

def analyze_env(env):
    if isinstance(env.observation_space, gym.spaces.Box):
        space_type = "Continuous"
    else:
        space_type = "Discrete"
    template = (" = Environment info =\n"
                "Observation space: {obs_space}\n"
                "\ttype: {space_type}\n")
    # print(template.format(
    #     obs_space=env.observation_space,
    #     space_type=space_type)
    # )

def alpha_decreased(x, max_y=1, min_y=0.3, smoothness=0.2):
    return min_y+math.exp(-smoothness*x)*(max_y-min_y)

def train_agent(env, env_eval, config=None, verbose=False, oneline=True, cyclic=False):
    times = dict(t_ini=time.time(), t_tr_tot=0, t_ev_tot=0)
    if config is None or isinstance(config, str):
        config = get_config(config)
    agent = get_agent_from_config(env=env, config=config, verbose=verbose)
    monitor = get_monitor_from_config(agent=agent, env_eval=env_eval, config=config)
    random.seed(config['seed'])
    epoch = 0
    ema = 0
    agent.fill_memory(env, cyclic=cyclic)
    h_I = np.zeros(agent.obs_shape, dtype=np.float32)
    steps_tot = 0
    monitor.evalue(steps_tot, times, verbose=verbose, oneline=oneline, h_I=h_I)
    while not monitor.stop:
        t_tr_s = time.time()
        if verbose and not oneline: t = tqdm.tqdm(total=ema)
        if cyclic:
            try:
                obs = env.reset(obs)
            except NameError:
                obs = env.reset()
                h = h_I
        else:
            obs = env.reset()
            h = np.zeros(agent.obs_shape, dtype=np.float32)
        done = False
        step = 0
        while not done:
            step += 1
            h = agent.guess(obs, h)
            exploration = random.random() < agent.explore_step()
            a = agent(h) if not exploration else agent.action_space.sample()
            obs, r, done, info = env.step(a)
            agent.add_experience(h, a, r, obs, done)
            monitor.add_loss(agent.train_step())
            monitor.add_experience(h, a, r, obs, done)
            if verbose and not oneline: t.update()
        if verbose and not oneline:
            n = t.n
            t.close()
            ema = int(round(ema + alpha_decreased(epoch) * (n - ema)))
        epoch += 1
        steps_tot += step
        times['t_tr_tot'] += time.time() - t_tr_s
        if cyclic:
            monitor.evalue(steps_tot, times, verbose=verbose, oneline=oneline, h_I=h_I)
        else:
            monitor.evalue(steps_tot, times, verbose=verbose, oneline=oneline, h_I=np.zeros(agent.obs_shape, dtype=np.float32))
    if verbose:
        print("===================== End =====================")
