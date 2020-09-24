#!/usr/bin/env python
from collections import namedtuple
from collections.abc import Iterable
import os.path
import argparse
import copy
import sys
import yaml
import gym
from .core import Agentv1, Agentv2, DQNv1, DQNv2, Memoryv1, Monitorv1
import numpy as np

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
    parser.add_argument("-w", "--windows_size", dest="history_window", type=int, default=None)
    parser.add_argument("-m", "--model-path", dest="model_path", type=str, default=None)
    parser.add_argument("-c", "--config-file-path", dest="config_filename", type=str, default=None)
    parser.add_argument("-s", "--seed", type=str, default=None)
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

def _update_config(config, args):
    # import ipdb; ipdb.set_trace()
    if args.history_window: config['agent']['history_window'] = args.history_window
    if args.model_path: config['resources']['running']['model_path'] = args.model_path
    if args.learning_rate: config['agent']['network']['learning_rate'] = args.learning_rate
    if args.seed is None: args.seed = np.random.randint(2**32)
    if isinstance(args.seed, str): args.seed = list(map(int, args.seed.strip("()").split(",")))
    if isinstance(args.seed, int): args.seed = [args.seed, args.seed, args.seed]
    if len(args.seed) == 1: args.seed = [args.seed[0], args.seed[0], args.seed[0]]
    print("Seed:",args.seed)
    config['seed'] = args.seed
    config['max_epochs'] = args.n_epochs
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
    
def get_agent(label, model, target, memory=None, config=None):
    """By default, load the default config"""
    Agent_class = {"Agentv1": Agentv1, "Agentv2": Agentv2}
    if memory is None:
        raise GetAgentError("Memory is None, call get_memory function to retrieve a memory, or insert a memory label")
    if config is None:
        config = get_config()
    if isinstance(memory, str):
        memory = get_memory(memory, config)
    try:
        agent = Agent_class[label](model=model, target=target, memory=memory, config=config)
    except KeyError as e:
        raise (LabelError("Label {} not found. Must be {}".format(e, list(Agent_class.keys()))
            ).with_traceback(sys.exc_info()[2]))
    return agent

class LabelError(KeyError):
    pass

def get_model(label, observation_space, action_space, name, config):
    DQN_class = {"DQNv1": DQNv1, "DQNv2": DQNv2}
    try:
        model = DQN_class[label](observation_space, action_space, name=name, config=config)
    except KeyError as e:
        raise (LabelError("Label {} not found. Must be {}".format(e, list(DQN_class.keys()))
            ).with_traceback(sys.exc_info()[2]))
    return model

def get_memory(label, config):
    return Memoryv1(
        max_size=config['agent']['memory']['max_size'],
        min_size=config['agent']['memory']['min_size'],
        seed=config['seed'][1])

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
