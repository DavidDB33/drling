import copy
import datetime
import json
import logging
logger = logging.getLogger(__file__)
import os
import os.path
import random
import statistics as stt
import sys
from collections import deque

import gym
import h5py
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except:
    print("Warning: matplotlib is not installed. You will not be able to display reward progress and can cause errors also")
    pass
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To silent Tensorflow warnings
from tqdm import autonotebook as tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D

class Memoryv1():
    def __init__(self, max_size = 100000, min_size = 1000, seed = None):
        self.buffer = deque(maxlen = max_size)
        self.rng = random.Random()
        if seed: self.rng.seed(seed)  # Better performance than self.rng = np.random.default_rng(seed)
        self.min_size = min_size

    def fill_memory(self, env, agent):
        s = env.reset()
        h = None
        for _ in tqdm.tqdm(range(self.min_size), desc="Memory"):
            h = agent.guess(s, h)
            a = env.action_space.sample()
            s, r, done, _ = env.step(a)
            agent.add_experience(h, a, r, s, done)
            if done:
                s = env.reset()
                h = None

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return self.rng.sample(self.buffer, batch_size)

class NNv1(Model):
    def __init__(self, n_output, **kwargs):
        super().__init__(**kwargs)
        self.d1 = Dense(256, activation='relu', kernel_initializer="he_uniform")
        self.d2 = Dense(64, activation='relu', kernel_initializer="he_uniform")
        self.d3 = Dense(16, activation='relu', kernel_initializer="he_uniform")
        self.d4 = Dense(n_output, activation=None)

    @tf.function
    def call(self, x):
        y = self.d1(x)
        y = self.d2(y)
        y = self.d3(y)
        y = self.d4(y)
        return y

class NNv2(Model):
    def __init__(self, n_output, **kwargs):
        super().__init__(**kwargs)
        self.c1 = Conv1D(16, 2, padding='valid', activation='relu', kernel_initializer="glorot_uniform", name="conv1d_1")
        self.c2 = Conv1D(16, 2, padding='valid', activation='relu', kernel_initializer="glorot_uniform", name="conv1d_2")
        self.f1 = Flatten(name="flatten_1")
        self.d1 = Dense(50, activation='relu', kernel_initializer="he_uniform", name="dense_1")
        self.d2 = Dense(20, activation='relu', kernel_initializer="he_uniform", name="dense_2")
        self.d3 = Dense(n_output, activation=None, kernel_initializer="he_uniform", name="dense_3")

    @tf.function
    def call(self, x):
        y = self.c1(x)
        y = self.c2(y)
        y = self.f1(y)
        y = self.d1(y)
        y = self.d2(y)
        y = self.d3(y)
        return y

class DQNv1():
    def __init__(self, observation_space, action_space, name, config):
        self.action_space = action_space
        self.window_size = (config['agent']['history_window'] if 'history_window' in config['agent'] and config['agent']['history_window'] is not None else 1,)
        self.obs_shape = self.window_size + observation_space.shape
        self.n_output = action_space.shape and np.product(action_space.nvec) or action_space.n
        self.loss_object = tf.keras.losses.MeanSquaredError() # Try huber loss
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=config['agent']['network']['learning_rate'])
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.nn = self._get_nn(self.n_output, name=name)
        self.gamma = config['agent']['network']['gamma']
        self.tf_gamma = tf.constant(self.gamma)
        self.load_weights = self.nn.load_weights
        self.save_weights = self.nn.save_weights

    def _get_nn(self, n_output, name):
        return NNv1(n_output, name=name)

    @tf.function
    def __call__(self, x):
        return self.nn(x)

    @tf.function
    def qvalue(self, x, a):
        x = self.nn(x)
        a = tf.one_hot(a, self.n_output)
        x = x * a
        x = tf.reduce_sum(x, axis=-1)
        return x

    @tf.function
    def qvalues(self, x):
        return self.nn(x)

    @tf.function
    def qvalue_max(self, x):
        x = self.nn(x)
        x = tf.reduce_max(x, axis=-1)
        return x

    @tf.function
    def argmax_qvalue(self, x):
        x = self(x)
        x = tf.argmax(x, axis=-1)
        return x

    @tf.function
    def qtarget(self, x, r):
        return r + self.gamma*self.qvalue_max(x)

class DQNv2(DQNv1):
    def _get_nn(self, n_output, name):
        return NNv2(n_output, name=name)

class Agentv1():
    def __init__(self, model, memory, config):
        self.action_space = model.action_space
        self.step = 0
        self.explore_start = config['agent']['explore_start']
        self.explore_stop = config['agent']['explore_stop']
        self.decay_rate = config['agent']['decay_rate']
        self.batch_size = config['agent']['network']['batch_size']
        self.history_window = config['agent']['history_window'] if 'history_window' in config['agent'] and config['agent']['history_window'] is not None else 1
        self.config = config
        self.model = model
        self.ndim = self._get_dimensions()
        # self.target = target
        self.memory = memory

    def __call__(self, *args, **kwargs):
        """Alias for self.act"""
        return self.act(*args, **kwargs)

    def _get_dimensions(self):
        return 2

    def fill_memory(self, env):
        self.memory.fill_memory(env, self)

    @property
    def obs_shape(self):
        return self.model.obs_shape

    def load_weights(self, path, load_from_path=True, skip_OSError=False):
        if not self.model.nn.built:
            shape_nn = (None,*self.obs_shape)
            self.model.nn.build(shape_nn)
            # self.target.nn.build(shape_nn)
        try:
            if load_from_path:
                self.model.load_weights(path)
        except OSError as e:
            print("Model not initialized. OSError skipped", file=sys.stderr)
            if not skip_OSError:
                raise e
        finally:
            pass
            # self.target.nn.set_weights(self.model.nn.get_weights())


    def act(self, belief, keep_tensor=False):
        """Perform an action. 
        If keep_tensor=True then return a tensor of Tensorflow.
        
        Args:
            belief (numpy.ndarray): Agent's belief of the env state given n observations (computed by agent.guess(o, h))

        Return:
            act ([numpy.array, tensorflow.tensor]): 
        """
        if belief.ndim < self.ndim:
            belief = belief[None, ...]
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            act = self.model.argmax_qvalue(belief)
            act = tf.unravel_index(act, self.action_space.nvec)
        else:
            raise NotImplementedError("Currently only implemented for DQN")
        if not keep_tensor:
            act = act.numpy()
        return act

    def qvalues(self, s):
        """Return qvalues given an state s"""
        return self.model(s)

    def explore(self, increment=False):
        """Exploration with Exponential Decaying:
        Description: Compute probability of exploring using p(ε) = stop + (start-stop)/exp(decay*step)
        Internal Params:
            start = 1.0 (float): At start only explore
            stop = 0.1 (float): Minimum exploration rate
            decay = 1e-4 (float): decay rate of probability
            step ∈ {[0, ∞) ∩ ℕ } starts in 0 and step++
        Args:
            increment (bool): If true, increment self.step in 1 automatically. Default: False
        Returns:
            explore_p (numpy.float): Probability of exploration ∈ (stop, start]
        """
        explore_p = self.explore_stop + np.exp(-self.decay_rate*self.step)*(self.explore_start - self.explore_stop)
        if increment:
            self.step += 1
        return explore_p  # Make a random action

    def explore_step(self):
        return self.explore(increment=True)

    def guess(self, obs = None, hstate = None):
        return obs

    def _ravel_action(self, action):
        if np.array(action).size > 1:
            action = np.ravel_multi_index(action, self.action_space.nvec).squeeze()
        return action

    def add_experience(self, obs, action, reward, next_obs, done):
        obs = np.array(obs)
        action = self._ravel_action(action)
        experience = obs, action, reward, next_obs, done
        self.memory.add(experience)

    def sample(self):
        action = self.action_space.sample()
        return action

    def train_step(self):
        experience_list = self.memory.sample(self.batch_size)
        hstate_list, action_list, reward_list, next_obs_list, done_list = list(zip(*experience_list))
        next_hstate_list = [self.guess(obs, hstate) for hstate, obs in zip(hstate_list, next_obs_list)]
        hstate_tensor = tf.convert_to_tensor(hstate_list, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action_list, dtype=tf.int32)
        reward_tensor = tf.convert_to_tensor(reward_list, dtype=tf.float32)
        next_hstate_tensor = tf.convert_to_tensor(next_hstate_list, dtype=tf.float32)
        # self.target.nn.set_weights(self.model.nn.get_weights())
        self.tf_train_step(hstate_tensor, action_tensor, reward_tensor, next_hstate_tensor) #, done_tensor)
        loss = self.model.train_loss.result().numpy()
        self.model.train_loss.reset_states()
        return loss

    @tf.function
    def tf_train_step(self, hstate_input, action_tensor, reward_tensor, next_hstate_tensor): #, done_list):
        """
            q(s,a)_t+1 = q(s,a) - α*err
            err = q(s,a) - r+γ*max_a[q(s',a)]
            Only compute error + SGD instead of computing moving average and then SGD
        """
        qtarget = self.model.qtarget(next_hstate_tensor, reward_tensor) # *(1-done_list) # r+γ*max_a[q(s',a')]
        model_variables = self.model.nn.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_variables)
            qvalue = self.model.qvalue(hstate_input, action_tensor) # q(s,a)
            loss = self.model.loss_object(qtarget, qvalue)
        gradients = tape.gradient(loss, model_variables)
        self.model.optimizer.apply_gradients(zip(gradients, model_variables))
        self.model.train_loss(loss)
        return gradients

class Agentv2(Agentv1):
    def _get_dimensions(self):
        return 3

    def guess(self, obs = None, hstate = None):
        if obs is None:
            return np.zeros(self.model.obs_shape, dtype=np.float32)
        if hstate is None:
            hstate = np.zeros(self.model.obs_shape, dtype=np.float32)
        hstate = np.roll(hstate, -1, axis=0)
        hstate[-1, ...] = obs
        return np.array(hstate)

class Monitorv1():
    def __init__(self, agent, env_eval, config):
        self.config = config
        self.agent = agent
        self.env_eval = env_eval
        self._done = False
        self.dt = datetime.datetime.now()
        # suffix = "_w{:02}".format(config['agent']['history_window']) if 'history_window' in config['agent'] and config['agent']['history_window'] is not None else ""
        # self.training_path = os.path.join(os.path.abspath(config['resources']['training']['results']), self.dt.strftime("%Y-%m-%dT%H-%M-%S") + suffix)
        # self.running_path = os.path.join(os.path.abspath(config['resources']['rl_model']['output']), self.dt.strftime("%Y-%m-%dT%H-%M-%S") + suffix)
        self.output_data = config['output']['data']
        self.output_model = config['output']['model']
        self.epoch = 0
        self.early_stop_iterations = 0
        self.early_stop_max_iterations = config['agent']['monitor']['early_stop']
        self.best_reward = None
        self.plot_epoch = 0
        self.plot_rewards = list()
        self.plot_displayed = None
        self.training_data = list()
        self.eval_data = list()
        self.loss_list = list()
        self.ema = 0

    def add_experience(self, s, a, r, s_, done):
        self.training_data.append((s, a, r, s_, done))

    def add_loss(self, loss):
        self.loss_list.append(loss)

    def _save_epoch(self, epoch):
        """Store in h5/epoch training, eval and loss data gather from self
        Args:
            epoch (str): Epoch that determine the new subgroup in h5
        """
        s_t_list, a_t_list, r_t_list, s__t_list, done_t_list = list(zip(*self.training_data))
        s_e_list, a_e_list, r_e_list, s__e_list, done_e_list = list(zip(*self.eval_data))
        data_dict = {
            '%s/t/s': s_t_list,
            '%s/t/a': a_t_list,
            '%s/t/r': r_t_list,
            '%s/t/s_': s__t_list,
            '%s/t/d': done_t_list,
            '%s/l': self.loss_list,
            '%s/e/s': s_e_list,
            '%s/e/a': a_e_list,
            '%s/e/r': r_e_list,
            '%s/e/s_': s__e_list,
            '%s/e/d': done_e_list,
        }
        with h5py.File(self.output_data, "a") as f:
            for k, v in data_dict.items():
                try:
                    f[k%epoch] = v
                except (OSError, RuntimeError):
                    f[k%epoch][...] = v

    def _save_model(self):
        """Save the model of the agent in self.output_model"""
        self.agent.model.nn.save_weights(self.output_model, save_format='h5')

    def _clear_data(self):
        """Clean all data stored from the epoch for the next train step"""
        self.training_data = list()
        self.eval_data = list()
        self.loss_list = list()

    def evalue(self):
        env = self.env_eval
        agent = self.agent
        obs = env.reset()
        h = None
        done = False
        reward_list = list()
        while not done:
            h = agent.guess(obs, h)
            action = agent(h)
            obs_next, reward, done, _ = env.step(action)
            self.eval_data.append((obs, action, reward, obs_next, done))
            reward_list.append(reward)
        total_reward = sum(reward_list)
        if self.best_reward is None or total_reward > self.best_reward:
            self.best_reward = total_reward
            self._save_model()
            self.early_stop_iterations = 0
        else:
            self.early_stop_iterations += 1
        print("Training summary (epoch {}):".format(self.epoch))
        print("  Training Mean loss: {}".format(np.array(self.loss_list).mean()))
        print("  Evaluation reward: {}".format(total_reward))
        print("  Early iter remaining: {}".format(self.early_stop_max_iterations - self.early_stop_iterations))
        self.epoch += 1
        self._save_epoch(str(self.epoch))
        self._clear_data()

    @property
    def stop(self):
        return self.early_stop_iterations >= self.early_stop_max_iterations
        
    @property
    def has_improved(self):
        return self._has_improved

    def training(self, training_results, developing_results = None):
        os.makedirs(self.training_path, exist_ok=True)
        self.epoch += 1

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
        self._has_improved = self.best_reward <= new_best_reward
        if self._has_improved:
            self.best_reward = new_best_reward
            self.early_stop = 0
        else:
            self.early_stop += 1
        self._done = self.early_stop >= self.config['monitor']['early_stop']

    def running(self, results):
        os.makedirs(self.running_path, exist_ok=True)
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
        print("\tDEBUG: Exploration: %f"%agent.explore())

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

