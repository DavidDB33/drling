#!/usr/bin/env python
import argparse
import copy
import logging
logger = logging.getLogger(__name__)
import math
import os.path
import random
import sys
import time
from collections import namedtuple
from collections.abc import Iterable
import yaml
import gym
import numpy as np
from tqdm import autonotebook as tqdm
import tensorflow as tf
from .core import Agentv1, Agentv2, DQN, Monitor
from .memory import Memoryv1

_config_file = None # Singleton
_config = None # Singleton
_parser = None # Singleton

class GetAgentError(Exception):
    pass

class LabelError(KeyError):
    pass

def update_config_with_args(config, args):
    if args.history_window is not None: config['agent']['history_window'] = args.history_window
    if args.model_path is not None: config['resources']['running']['model_path'] = args.model_path
    if args.learning_rate is not None: config['agent']['network']['learning_rate'] = args.learning_rate
    if args.seed is not None: config['seed'] = args.seed
    if args.n_epochs is not None: config['max_epochs'] = args.n_epochs
    return config

def get_seed():
    return None

def load_config_from_file(path = None):
    if path is None:
        filename = os.path.join("config", "agent.config.yaml")
    else:
        filename = os.path.expanduser(path)
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    if 'seed' not in config or config['seed'] is None: config['seed'] = get_seed()
    return config

def load_config(path = None):
    """If path is specified, returns the config from path, else get a copy of the last config loaded as a singleton.

    Program parameters have priority over the file
    Args:
        path (str): Path to the configuration file. If path is None and a config was never loaded, use default of load_config_from_file()

    Returns:
        _config (dict): A dict with the file config updated by program params
    """
    global _config
    if path is None and _config is not None:
        return copy.deepcopy(_config)
    _config = load_config_from_file(path)
    return copy.deepcopy(_config)

def set_config(config):
    global _config
    _config = config
    
def get_agent(label, model, seed=None, config=None, verbose=False):
    """By default, load the default config"""
    Agent_class = {"Agentv1": Agentv1, "Agentv2": Agentv2}
    if config is None:
        config = load_config()
    memory = Memoryv1(
        max_size=config['agent']['memory']['max_size'],
        min_size=config['agent']['memory']['min_size'],
        seed=seed,
        verbose=verbose)
    try:
        # print("LOAD AGENT: {}".format(label))
        agent = Agent_class[label](model=model, memory=memory, config=config)
    except KeyError as e:
        raise (LabelError("Label {} not found. Must be {}".format(e, list(Agent_class.keys()))
            ).with_traceback(sys.exc_info()[2]))
    return agent

def get_agent_from_config(env, config, seed, verbose=False):
    """Maybe the most interesing method in this file"""
    # model = get_model(label=config['agent']['network']['name'], observation_space=env.observation_space, action_space=env.action_space, config=config)
    if isinstance(config, str):
        config = load_config(config)
    model = DQN(env.observation_space, env.action_space, name='DQN', config=config)
    agent = get_agent(label=config['agent']['name'], model=model, seed=seed, config=config, verbose=verbose)
    return agent

def analyze_env(env):
    if isinstance(env.observation_space, gym.spaces.Box):
        space_type = "Continuous"
    else:
        space_type = "Discrete"
    template = (" = Environment info =\n"
                "Observation space: {obs_space}\n"
                "\ttype: {space_type}\n")
    print(template.format(
        obs_space=env.observation_space,
        space_type=space_type)
    )

def alpha_decreased(x, max_y=1, min_y=0.3, smoothness=0.2):
    return min_y+math.exp(-smoothness*x)*(max_y-min_y)

def train_agent(env, env_eval_list, conf_name=None, verbose=False, oneline=True, cyclic=False, seed=None, output_template=None, max_steps=sys.maxsize):
    if seed is not None:
        logger.info("Seed: {}".format(seed))
        tf.random.set_seed(seed)
        random.seed(seed)
        env.seed(seed)
        for env_eval in env_eval_list:
            env_eval.seed(seed)
    n_train_steps = 5 # 999999
    eval_step_each = 24
    times = {
        't_ini': time.time(),
        't_ema_step': 0,
        't_last_step': time.time(),
        't_step_delta_list': [],
        't_tr_tot': 0,
        't_ev_tot': 0
    }
    if conf_name is None or isinstance(conf_name, str):
        config = load_config(conf_name)
    agent = get_agent_from_config(env=env, config=config, seed=seed, verbose=verbose)
    monitor = Monitor(agent=agent, env_eval_list=env_eval_list, config=config, times=times, output_template=output_template)
    epoch = 0
    ema = 0
    step_eval = 0
    continuous = config['agent'].get("continuous", False)
    agent.fill_memory(env, cyclic=cyclic)
    h_I = None
    steps_tot = 0
    monitor.evalue(steps_tot, epoch, verbose=verbose, oneline=oneline, h_I=None)
    t_tr_s = time.time()
    while not monitor.stop and max_steps > steps_tot:
        if verbose and not oneline: t = tqdm.tqdm(total=ema)
        if cyclic:
            try:
                obs = env.reset(obs)
            except NameError:
                obs = env.reset()
                h = np.zeros(agent.obs_shape, dtype=np.float32)
        else:
            obs = env.reset()
            h = np.zeros(agent.obs_shape, dtype=np.float32)
        done = False
        step = 0
        while not done:
            h = agent.guess(obs, h)
            exploration = random.random() < agent.explore_step()
            a = agent(h) if not exploration else agent.action_space.sample()
            obs, r, done, info = env.step(a)
            agent.add_experience(h, a, r, obs, done and not continuous) # Continuous environment
            for ts in range(n_train_steps):
                monitor.add_loss(agent.train_step())
                step += 1
                steps_tot += 1
                if steps_tot >= (step_eval + 1) * eval_step_each:
                    times['t_tr_tot'] += time.time() - t_tr_s
                    monitor.evalue(steps_tot, epoch, verbose=verbose, oneline=oneline, h_I=None)
                    step_eval += 1
                    t_tr_s = time.time()
            monitor.add_experience(h, a, r, obs, done)
            time_now = time.time()
            times['t_step_delta_list'].append((n_train_steps/(time_now-times['t_last_step'])))
            times['t_last_step'] = time_now
            if verbose and not oneline: t.update()
            # Check to eval
            if steps_tot >= (step_eval + 1) * eval_step_each:
                times['t_tr_tot'] += time.time() - t_tr_s
                monitor.evalue(steps_tot, epoch, verbose=verbose, oneline=oneline, h_I=None)
                step_eval += 1
                t_tr_s = time.time()
            if monitor.stop:
                break
        if verbose and not oneline:
            n = t.n
            t.close()
            ema = int(round(ema + alpha_decreased(epoch) * (n - ema)))
        epoch += 1
    if verbose:
        print("===================== End =====================", flush=True)
