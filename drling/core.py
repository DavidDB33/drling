import sys
import os
import copy
import datetime
import json
import os.path
import statistics as stt
from collections import deque
from functools import reduce, partial
from operator import mul
import gym
prod = partial(reduce, mul)
try:
    import matplotlib.pyplot as plt
except:
    print("Warning: matplotlib is not installed. You will not be able to display reward progress and can cause errors also")
    pass
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Memoryv1():
    def __init__(self, size = 100000):
        self.buffer = deque(maxlen = size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size = batch_size, replace = False)
        return [self.buffer[ii] for ii in idx]

class DQNv1(Model):
    def __init__(self, action_space, observation_space, config):
        super(DQNv1, self).__init__()
        self.action_space = action_space
        self.window_size = (config['agent']['history_window'] if 'history_window' in config['agent'] and config['agent']['history_window'] is not None else 1,)
        self.obs_shape = observation_space.shape + self.window_size
        self.n_output = action_space.shape and prod(action_space.nvec) or action_space.n
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=config['agent']['network']['learning_rate'])# 0.001)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self._make(config)
        self.compile()
        # self(tf.convert_to_tensor(np.expand_dims(np.zeros(self.obs_shape), 0), dtype=tf.float32))

    def _make(self, config):
        self.d1 = Dense(256, activation='relu', kernel_initializer="he_uniform")
        self.d2 = Dense(64, activation='relu', kernel_initializer="he_uniform")
        self.d3 = Dense(16, activation='relu', kernel_initializer="he_uniform")
        self.d4 = Dense(self.n_output, activation=None)

    @tf.function
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

class DQNv2(DQNv1):
    def _make(self, config):
        self.c1 = Conv1D(16, 2, padding='valid', activation='relu', kernel_initializer="glorot_uniform")
        self.c2 = Conv1D(16, 2, padding='valid', activation='relu', kernel_initializer="glorot_uniform")
        self.f1 = Flatten()
        self.d1 = Dense(50, activation='relu', kernel_initializer="he_uniform")
        self.d2 = Dense(20, activation='relu', kernel_initializer="he_uniform")
        self.d3 = Dense(self.n_output, activation=None, kernel_initializer="he_uniform")

    @tf.function
    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x

class Agentv1():
    def __init__(self, model, memory, config):
        self.action_space = model.action_space
        self.alpha = .1 # UNUSED
        self.gamma = config['agent']['network']['gamma']
        self.step = 0
        self.explore_start = config['agent']['explore_start']
        self.explore_stop = config['agent']['explore_stop']
        self.decay_rate = config['agent']['decay_rate']
        self.batch_size = config['agent']['network']['batch_size']
        self.history_window = config['agent']['history_window'] if 'history_window' in config['agent'] and config['agent']['history_window'] is not None else 1
        self.config = config
        self.model = model
        self.memory = memory
        self.qvalue = lambda a, x: tf.reduce_sum(x*tf.one_hot(a, model.n_output), axis=-1)
        self.max_qvalue = lambda x: tf.reduce_max(x, axis=-1)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @property
    def n_obs(self):
        return self._n_obs

    @n_obs.setter
    def n_obs(self, value):
        self._n_obs = value.shape and prod(value.shape) or value.n

    def load_weights(self, path, skip_OSError=False):
        try:
            if not self.model.built:
                belief = self.guess()
                belief_nn = np.expand_dims(belief, axis=0)
                self.model.build(belief_nn.shape)
            self.model.load_weights(path)
        except OSError as e:
            print("Model not initialized. OSError skipped", file=sys.stder)
            if not skip_OSError:
                raise e

    def act(self, x):
        act = tf.argmax(x, axis=-1)
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return np.unravel_index(act, self.action_space.nvec)
        else:
            return act

    def call(self, obs):
        return self.model(tf.convert_to_tensor(obs, dtype=tf.float32))

    def explore(self, increment=False):
        """
        Exploration with Exponential Decaying:
            Description: Compute probability of exploring
            Formula: p(ε) = stop + (start-stop)/exp(decay*step)
            Example values:
                start = 1.0 (At start only explore)
                stop = 0.1 (Minimum exploration rate)
                decay = 1e-4
                step starts in 0 and step++
        """
        explore_p = self.explore_stop + np.exp(-self.decay_rate*self.step)*(self.explore_start - self.explore_stop)
        if increment:
            self.step += 1
        return explore_p  # Make a random action

    def explore_step(self):
        return self.explore(increment=True)

    def guess(self, obs = None, hstate = None):
        return obs

    def add_experience(self, obs, action, reward, next_obs, done):
        obs = np.array(obs)
        if obs.shape[-1] == 1:
            obs = obs.reshape(obs.shape[:-1])
        if np.array(action).size > 1:
            action = np.ravel_multi_index(action, self.action_space.nvec)
        experience = obs, action, reward, next_obs, done
        self.memory.add(experience)

    def sample(self):
        action = self.action_space.sample()
        return action

    def train_step(self):
        experiences = self.memory.sample(self.batch_size)
        hstate_list, action_list, reward_list, next_obs_list, done_list = list(zip(*experiences))
        next_hstate_list = np.array([self.guess(obs, hstate) for hstate, obs in zip(hstate_list, next_obs_list)])
        hstate_tensor = tf.convert_to_tensor(hstate_list, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action_list, dtype=tf.int32)
        reward_tensor = tf.convert_to_tensor(reward_list, dtype=tf.float32)
        next_hstate_tensor = tf.convert_to_tensor(next_hstate_list, dtype=tf.float32)
        done_tensor = tf.convert_to_tensor(done_list, dtype=tf.float32)
        self.tf_train_step(hstate_tensor, action_tensor, reward_tensor, next_hstate_tensor, done_tensor)
        loss = self.model.train_loss.result().numpy()
        self.model.train_loss.reset_states()
        return loss

    @tf.function
    def tf_train_step(self, obs, actions, rewards, next_obs, done_list):
        """
            q(s,a)_t+1 = q(s,a) - α*err
            err = q(s,a) - r+γ*max_a[q(s',a)]
            Only compute error + SGD instead of computing moving average and then SGD
        """
        new_expected_rewards = rewards + (1-done_list)*self.gamma*self.max_qvalue(self(next_obs)) # r+γ*max_a[q(s',a')]
        with tf.GradientTape() as tape:
            expected_rewards = self.qvalue(actions, self.model(obs)) # q(s,a)
            # q_value = expected_rewards - self.alpha*(expected_rewards - new_expected_rewards)
            loss = self.model.loss_object(new_expected_rewards, expected_rewards)
            # loss = self.model.loss_object(q_value, expected_rewards) # , new_expected_rewards, expected_rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.model.train_loss(loss)

class Agentv2(Agentv1):
    def guess(self, obs = None, hstate = None):
        if obs is None:
            return np.zeros((self.history_window, self._n_obs))
        hstate = np.roll(np.array(hstate), -1, axis=0)
        hstate[-1, ...] = obs
        return np.array(hstate)

class Monitorv1():
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.trainer_data = []
        self.developer_data = []
        self._done = False
        self.dt = datetime.datetime.now()
        suffix = "_w{:02}".format(config['agent']['history_window']) if 'history_window' in config['agent'] and config['agent']['history_window'] is not None else ""
        self.training_path = os.path.join(config['resources']['training']['results'], self.dt.isoformat(timespec='seconds') + suffix)
        self.running_path = os.path.join(config['resources']['rl_model']['output'], self.dt.isoformat(timespec='seconds') + suffix)
        self.epoch = 0
        self.early_stop = 0
        self.best_reward = None
        self.plot_epoch = 0
        self.plot_rewards = list()
        self.plot_displayed = None
        # os.mkdir(self.training_path)
        # os.mkdir(self.running_path)

    @property
    def done(self):
        return self._done
        
    @property
    def improved(self):
        return self._improved

    def training(self, training_results, developing_results = None):
        self.epoch += 1

        os.makedirs(self.training_path, exist_ok=True)
        os.makedirs(self.running_path, exist_ok=True)

        ### Training
        obs_list, action_list, reward_list, loss_list = training_results

        template = "Training: epoch #{} total reward {}, mean loss {}"
        print(template.format(self.epoch, sum(reward_list), stt.mean(loss_list)))

        ### Developing
        obs_dev_list, action_dev_list, reward_dev_list = developing_results
        template_dev = "Developing: epoch #{} total reward {}"
        print(template_dev.format(self.epoch, sum(reward_dev_list)))

        ### Write results
        obs_np = np.array(obs_list)
        obs_dev_np = np.array(obs_dev_list)
        action_np = np.array(action_list).reshape((obs_np.shape[0],-1))
        action_dev_np = np.array(action_dev_list).reshape((obs_np.shape[0],-1))
        reward_np = np.expand_dims(reward_list, axis=-1)
        loss_np = np.expand_dims(loss_list, axis=-1)
        reward_dev_np = np.expand_dims(reward_dev_list, axis=-1)
        train_data = np.concatenate([obs_np, action_np, reward_np, loss_np], axis=-1)
        dev_data = np.concatenate([obs_dev_np, action_dev_np, reward_dev_np], axis=-1)
        obs_columns = ["obs_%i"%i for i in range(np.array(obs_list).shape[-1])]
        obs_dev_columns = ["obs_dev_%i"%i for i in range(np.array(obs_list).shape[-1])]
        action_columns = ["act_%i"%i for i in range(action_np.shape[-1])]
        action_dev_columns = ["act_dev_%i"%i for i in range(action_np.shape[-1])]
        train_columns = obs_columns + action_columns + ["reward", "loss"]
        dev_columns = obs_dev_columns + action_dev_columns + ["reward_dev"]
        df_train = pd.DataFrame(data=train_data, columns=train_columns)
        df_dev = pd.DataFrame(data=dev_data, columns=dev_columns)
        df_train.to_csv(os.path.join(self.training_path, "epoch%04d.train.csv"%self.epoch))
        df_dev.to_csv(os.path.join(self.training_path, "epoch%04d.dev.csv"%self.epoch))

        ### Early stop
        new_best_reward = sum(reward_dev_list)
        if self.best_reward is None:
            self.best_reward = new_best_reward
        self._improved = self.best_reward <= new_best_reward
        if self._improved:
            self.best_reward = new_best_reward
            self.early_stop = 0
        else:
            self.early_stop += 1
        self._done = self.early_stop >= self.config['monitor']['early_stop']

    def running(self, results):
        obs_list, action_list, reward_list = results
        obs_np = np.array(obs_list)
        action_np, reward_np = np.array(action_list).reshape((obs_np.shape[0],-1)), np.expand_dims(reward_list, axis=-1)
        data = np.concatenate([obs_np, action_np, reward_np], axis=-1)
        obs_columns = ["obs_%i"%i for i in range(obs_np.shape[-1])]
        action_columns = ["act_%i"%i for i in range(action_np.shape[-1])]
        columns = obs_columns + action_columns + ["reward"]
        df = pd.DataFrame(data=data, columns=columns)
        os.makedirs(self.running_path, exist_ok=True)
        df.to_csv(os.path.join(self.running_path, "results.csv"))
        reward = sum(reward_list)
        r_len = len(reward_list)
        rew1, rew2, rew3 = sum(reward_list[:r_len//3]), sum(reward_list[r_len//3:2*r_len//3]), sum(reward_list[2*r_len//3:])
        
        print("RL Agent Info:")
        print(f"Reward: {reward}")
        print(f"Rewards: {rew1} - {rew2} - {rew3}")
    
    def debug(self, agent):
        print("\tExploration: %f"%agent.explore())

    def plot(self, results):
        obs_list, action_list, reward_list = results
        new_reward = sum(reward_list)
        self.plot_rewards.append(new_reward)
        if self.plot_epoch == 0:
            plt.ion()
            plt.clf()
            self.epoch += 1
            self.last_reward = 0
        p = self.plot_epoch
        plt.plot([p, p+1], [self.last_reward, new_reward], color='b')
        plt.show()
        plt.pause(0.00001)
        self.plot_epoch += 1
        self.last_reward = new_reward

