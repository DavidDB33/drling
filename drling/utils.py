#!/usr/bin/env python
from collections import namedtuple
from collections.abc import Iterable
import os.path
import argparse
import copy
import sys
import numpy as np
import yaml
import gym
from .core import Agentv1, Agentv2, DQNv1, DQNv2, Memoryv1, Monitorv1

_config_file = None # Singleton
_config = None # Singleton
_parser = None # Singleton

class GetAgentError(Exception):
    pass

def set_parser(parser):
    global _parser
    _parser = parser

def get_parser(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Drling framework")
    subparsers = parser.add_subparsers()
    parser.add_argument("-w", type=int, dest="history_window")
    parser.add_argument("-m", type=str, dest="running_model_path")
    # parser.add_argument("--drling-config", type=str, dest="drling_config_path")
    train_parser = subparsers.add_parser('train', help="Train the agent")
    train_parser.set_defaults(exec="train")
    run_parser = subparsers.add_parser('run', help="Run the agent")
    run_parser.set_defaults(exec="run")
    defaults = dict(
        exec="run",
        history_window=None,
        model_path=None,
        config_path=None
    )
    parser.set_defaults(**defaults)
    return parser

def _update_config(config, args):
    if args.history_window:
        config['agent']['history_window'] = args.history_window
    if args.running_model_path:
        config['resources']['running']['model_path'] = args.running_model_path
    return config

def _get_file_config(path = None):
    global _config_file
    if _config_file is not None:
        return copy.deepcopy(_config_file)
    if path is None:
        config_file = "config.yaml"
    else:
        config_file = os.path.expanduser(path)
    with open(config_file, "r") as f:
        file_config_dict = yaml.safe_load(f)
    if __debug__:
        pass
        # print("========== CONFIG ==========", file=sys.stderr)
        # print(f"path: {os.path.abspath(config_file)}", file=sys.stderr)
        # print("------- CONFIG ==========", file=sys.stderr)
        # print("  "+"\n  ".join(yaml.dump(file_config_dict).strip().split('\n')), file=sys.stderr)
        # print("============================", file=sys.stderr)
    _config_file = file_config_dict
    return copy.deepcopy(_config_file)

def get_config(path = None):
    """Returns a copy of a singleton config dict.

    Program parameters have priority over the file
    Args:
        path (str): Path to the configuration file, default: microgrid/config.yaml

    Returns:
        _config (dict): A dict with the file config updated by program params
    """
    global _config
    if _config is not None:
        return copy.deepcopy(_config)
    _config_file = _get_file_config(path)
    if _parser is not None:
        args, _ = _parser.parse_known_args()
        _config_file = _update_config(_config_file, args)
    _config = _config_file
    return copy.deepcopy(_config)
    
def get_agent(label, model, memory=None, config=None):
    """By default, load the default config"""
    Agent_class = {"Agentv1": Agentv1, "Agentv2": Agentv2}
    if memory is None:
        raise GetAgentError("Memory is None, call get_memory function to retrieve a memory, or insert a memory label")
    if config is None:
        config = get_config()
    if isinstance(memory, str):
        memory = get_memory(memory, config)
    try:
        agent = Agent_class[label](model=model, memory=memory, config=config)
    except KeyError as e:
        raise (LabelError("Label {} not found. Must be {}".format(e, list(Agent_class.keys()))
            ).with_traceback(sys.exc_info()[2]))
    return agent

class LabelError(KeyError):
    pass

def get_model(label, observation_space, action_space, config):
    DQN_class = {"DQNv1": DQNv1, "DQNv2": DQNv2}
    try:
        model = DQN_class[label](observation_space, action_space, config)
    except KeyError as e:
        raise (LabelError("Label {} not found. Must be {}".format(e, list(DQN_class.keys()))
            ).with_traceback(sys.exc_info()[2]))
    # obs = np.expand_dims(model.obs2nn(observation_space.sample()), axis = 0)
    # model(obs)
    return model

def get_memory(label, config):
    return Memoryv1(config['agent']['memory']['size'])

def get_monitor(label, observation_space, action_space, config):
    return Monitorv1(observation_space, action_space, config)

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
