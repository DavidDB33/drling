#!/usr/bin/env python
from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
import argparse
import copy
import sys
import numpy as np
import yaml
import gym
from drling.core import Agentv1, Agentv2, DQNv1, DQNv2, Memoryv1, Monitorv1

_config = None # Singleton
_arguments = None # Singleton
_parser = None # Singleton

class GetAgentError(Exception):
    pass

def get_nested_namedtuple(config_dict, name = 'Config'):
    config_namedtuple = copy.deepcopy(config_dict)
    Config = namedtuple(name, config_dict.keys())
    for k, v in config_dict.items():
        if isinstance(config_dict[k], dict):
            config_namedtuple[k] = get_nested_namedtuple(v, k)
    config = Config(**config_namedtuple)
    return config

def parse_arguments():
    global _arguments
    global _parser
    if _arguments is not None:
        return _arguments, _parser
    parser = argparse.ArgumentParser(description="Drling framework")
    subparsers = parser.add_subparsers()
    parser.add_argument("-w", type=int, dest="history_window")
    parser.add_argument("-m", type=str, dest="model_path")
    train_parser = subparsers.add_parser('train', help="Train the agent")
    train_parser.set_defaults(exec="train")
    run_parser = subparsers.add_parser('run', help="Run the agent")
    run_parser.set_defaults(exec="run")
    parser.set_defaults(exec="run", history_window=None, model_path=None)
    args, unknown_args = parser.parse_known_args()
    print("ARGS:\n",args)
    _arguments, _parser = args, parser
    return args, parser

def _update_config(args, config):
    if args.history_window:
        config['agent']['history_window'] = args.history_window
    if args.model_path:
        config['resources']['training']['results'] = args.model_path

def get_config(path = None):
    global _config
    if _config is not None:
        return copy.deepcopy(_config)
    if path is None:
        config_file = Path("config.yaml")
    else:
        config_file = Path(path)
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    args, _ = parse_arguments()
    _update_config(args, config_dict)
    if __debug__:
        from pprint import pprint
        print("========== CONFIG ==========", file=sys.stderr)
        print("  "+"\n  ".join(yaml.dump(config_dict).strip().split('\n')), file=sys.stderr)
        print("============================", file=sys.stderr)
    _config = get_nested_namedtuple(config_dict)
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
        model = DQN_class[label](action_space, observation_space, config)
    except KeyError as e:
        raise (LabelError("Label {} not found. Must be {}".format(e, list(DQN_class.keys()))
            ).with_traceback(sys.exc_info()[2]))
    # obs = np.expand_dims(model.obs2nn(observation_space.sample()), axis = 0)
    # model(obs)
    return model

def get_memory(label, config):
    return Memoryv1(config.agent.memory.size)

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
    print(template.format(
        obs_space=env.observation_space,
        space_type=space_type)
    )
