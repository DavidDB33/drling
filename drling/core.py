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
import time
from collections import deque, namedtuple

from colorama import init, deinit, reinit, Fore, Style
init()
deinit()
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
from tensorflow.keras import Model, layers as Layers
from tensorflow.keras.layers import Dense, Flatten, Conv1D
# import tensorflow_addons as tfa

class Memoryv1():
    def __init__(self, max_size=100000, min_size=1000, seed=None, verbose=False):
        self.buffer = deque(maxlen = max_size)
        self.rng = random.Random()
        self.min_size = min_size
        self.min_size_iterable = self._get_min_size_iterable(min_size, verbose)
        if seed: self.rng.seed(seed)  # Better performance than self.rng = np.random.default_rng(seed)

    def _get_min_size_iterable(self, min_size, verbose):
        if verbose:
            iterable = tqdm.tqdm(range(self.min_size), desc="Memory")
        else:
            iterable = range(self.min_size)
        return iterable

    def fill_memory(self, env, agent, cyclic=False):
        o = env.reset()
        h = np.zeros(agent.obs_shape, dtype=np.float32)
        for _ in self.min_size_iterable:
            h = agent.guess(o, h)
            a = env.action_space.sample()
            o, r, done, _ = env.step(a)
            agent.add_experience(h, a, r, o, done)
            if done:
                if cyclic:
                    env.reset(o)
                else:
                    o = env.reset()
                    h = np.zeros(agent.obs_shape, dtype=np.float32)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return self.rng.sample(self.buffer, batch_size)

class NNv0(Model):
    def __init__(self, n_output, hidden_layers, **kwargs):
        super().__init__(**kwargs)
        self.nnlayers = list()
        for layer in hidden_layers:
            self.nnlayers.append(getattr(Layers, layer['class'])(*layer['args'], **layer['kwargs']))
        self.output_layer = Dense(n_output, activation=None)

    @tf.function
    def call(self, x):
        y = x
        for layer in self.nnlayers:
            y = layer(y)
        output = self.output_layer(y)
        return output

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

class DQNv0():
    def __init__(self, observation_space, action_space, name, config):
        self.action_space = action_space
        self.window_size = (config['agent']['history_window'] if 'history_window' in config['agent'] and config['agent']['history_window'] is not None else 1,)
        self.obs_shape = self.window_size + observation_space.shape
        self.n_output = action_space.shape and np.product(action_space.nvec) or action_space.n
        self.loss_object = tf.keras.losses.Huber()# MeanSquaredError() # Try huber loss
        self.learning_rate_initial = config['agent']['network']['learning_rate']
        self.learning_rate_schedule = self._get_learning_rate_schedule(config=config)
        # self.optimizer = tf.keras.optimizers.Nadam(learning_rate=config['agent']['network']['learning_rate'], clipnorm=1.0)
        # self.optimizer = tfa.optimizers.AdamW(learning_rate=config['agent']['network']['learning_rate'], weight_decay=self._get_wd(), clipvalue=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule, clipvalue=1.0)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.nn = self._get_nn(self.n_output, config=config, name=name)
        self.nn_target = self._get_nn(self.n_output, config=config, name=name)
        self.gamma = config['agent']['network']['gamma']

    def _get_learning_rate_schedule(self, config):
        # learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     learning_rate_initial,
        #     decay_steps=24*10000,
        #     decay_rate=0.1,
        #     staircase=True
        # ) # Same as piecewise 24*[10000, 20000, 30000, ...] [1e-3, 1e-4, 1e-5, ...] being lr_ini := 1e-3
        learning_rate_initial = config['agent']['network']['learning_rate'] # Repeated but maybe here is better
        if 'optimizer' in config: # Now only PiecewiseConstantDecay only available
            cfgopt = config['optimizer']
            bem = cfgopt['boundary epoch multiplier'] # This is not suitable (use better length of the episode)
            lri = learning_rate_initial
            learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[b*bem for b in cfgopt['boundaries']],
                values=[float(s)*lri for s in cfgopt['values']]
            )
        else: # Default and deprecated TODO: raise warning message
            learning_rate_schedule = learning_rate_initial
        return learning_rate_schedule

    def _build(self, shape_nn):
        self.nn.build(shape_nn)
        if not self.nn_target.built:
            self.nn_target.build(shape_nn)

    def _get_nn(self, n_output, config, name):
        if "name" in config['model']:
            name = config['model']['name']
        return NNv0(n_output, hidden_layers=config['model']['hidden_layers'], name=name)

    def load_weights(self, *args, **kwargs):
        ret = self.nn.load_weights(*args, **kwargs)
        self.nn_target.set_weights(self.nn.get_weights())
        return ret

    def save_weights(self, *args, **kwargs):
        return self.nn.save_weights(*args, **kwargs)

    @tf.function
    def __call__(self, x):
        return self.nn(x)

    @tf.function
    def qvalue(self, x, a):
        x = self.nn(x)
        a = tf.one_hot(a, self.n_output)
        x = x * a
        x = tf.reduce_sum(x, axis=1)
        return x

    @tf.function
    def qvalue_with_mask(self, x, mask):
        x = self.nn(x)
        x = tf.multiply(x, mask)
        x = tf.reduce_sum(x, axis=1)
        return x

    @tf.function
    def qvalues(self, x):
        return self.nn(x)

    @tf.function
    def qvalue_max(self, x):
        x = self.nn(x)
        x = tf.reduce_max(x, axis=1)
        return x

    @tf.function
    def target_qvalue_max(self, x):
        x = self.nn_target(x)
        x = tf.reduce_max(x, axis=1)
        return x

    @tf.function
    def argmax_qvalue(self, x):
        x = self.nn(x)
        x = tf.argmax(x, axis=1)
        return x

    @tf.function
    def qtarget(self, x, r, d):
        return r + self.gamma*self.target_qvalue_max(x)*(1-d)

    @tf.function
    def compute_error(self, o, a, r, n_o, done):
        return tf.keras.losses.MSE(self.qtarget(n_o, r), self.qvalue(o, a))

class DQNv1(DQNv0):
    def _get_nn(self, n_output, config, name):
        return NNv1(n_output, name=name)

class DQNv2(DQNv0):
    def _get_nn(self, n_output, config, name):
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
        self.seed = config['seed']
        self.config = config
        self.model = model
        self.ndim = self._get_dimensions()
        self.memory = memory
        self.train_steps_without_update = 0
        self.train_steps_to_update = 1

    def __call__(self, *args, **kwargs):
        """Alias for self.act"""
        return self.act(*args, **kwargs)

    def _get_dimensions(self):
        return 2

    @property
    def obs_shape(self):
        return self.model.obs_shape

    def fill_memory(self, env, cyclic=False):
        self.memory.fill_memory(env, self, cyclic=False)

    def load_weights(self, path, load_from_path=True, skip_OSError=False):
        if not self.model.nn.built:
            shape_nn = (None,*self.obs_shape)
            self.model._build(shape_nn)
        try:
            if load_from_path:
                self.model.load_weights(path)
        except OSError as e:
            if not skip_OSError:
                raise e
            print("Model not initialized. OSError skipped", file=sys.stderr)

    def act(self, belief, keep_tensor=False):
        """Perform an action. 
        If keep_tensor=True then return a tensor of Tensorflow.
        
        Args:
            belief (numpy.ndarray): Agent's belief of the env state given n observations (computed by agent.guess(o, h))

        Return:
            act ([numpy.array, tensorflow.tensor]): 
        """
        single_input_flag = False
        if belief.ndim < self.ndim:
            single_input_flag = True
            belief = belief[None, ...]
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            act = self.model.argmax_qvalue(belief)
            act = tf.unravel_index(act, self.action_space.nvec)
            act = tf.transpose(act)
            if single_input_flag:
                act = act[0]
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

    def sample(self):
        action = self.action_space.sample()
        return action

    def _ravel_action(self, action):
        if np.array(action).size > 1:
            action = np.ravel_multi_index(action, self.action_space.nvec).squeeze()
        return action

    def add_experience(self, obs, action, reward, next_obs, done):
        obs = np.array(obs)
        action = self._ravel_action(action)
        experience = obs, action, reward, next_obs, done
        self.memory.add(experience)

    def train_step(self):
        experience_list = self.memory.sample(self.batch_size)
        hstate_list, action_list, reward_list, next_obs_list, done_list = list(zip(*experience_list))
        next_hstate_list = [self.guess(obs, hstate) for hstate, obs in zip(hstate_list, next_obs_list)]
        hstate_tensor = tf.convert_to_tensor(hstate_list, dtype=tf.float32)
        action_tensor = tf.squeeze(tf.convert_to_tensor(action_list, dtype=tf.int32))
        reward_tensor = tf.convert_to_tensor(reward_list, dtype=tf.float32)
        next_hstate_tensor = tf.convert_to_tensor(next_hstate_list, dtype=tf.float32)
        done_tensor = tf.convert_to_tensor(done_list, dtype=tf.float32)
        if self.train_steps_without_update > self.train_steps_to_update:
            self.model.nn_target.set_weights(self.model.nn.get_weights())
            self.train_steps_without_update = 0
            if self.train_steps_to_update < 10000:
                self.train_steps_to_update += 1
        else:
            self.train_steps_without_update += 1
        self.tf_train_step(hstate_tensor, action_tensor, reward_tensor, next_hstate_tensor, done_tensor)
        loss = self.model.train_loss.result().numpy()
        self.model.train_loss.reset_states()
        # print(self.model.optimizer.lr)
        return loss

    @tf.function
    def tf_train_step(self, hstate_tensor, action_tensor, reward_tensor, next_hstate_tensor, done_tensor):
        """
            q(s,a)_t+1 = q(s,a) - α*err
            err = q(s,a) - r+γ*max_a[q(s',a)]
            Only compute error + SGD instead of computing moving average and then SGD
        """
        qtarget = self.model.qtarget(next_hstate_tensor, reward_tensor, done_tensor) # r+γ*max_a[q(s',a')]*(1-done)
        mask = tf.one_hot(action_tensor, self.model.n_output)
        with tf.GradientTape() as tape:
            qvalue = self.model.qvalue_with_mask(hstate_tensor, mask) # q(s,a)
            # qvalue = self.model.qvalue(hstate_tensor, action_tensor) # q(s,a)
            loss = self.model.loss_object(qtarget, qvalue)
        gradients = tape.gradient(loss, self.model.nn.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.nn.trainable_variables))
        self.model.train_loss(loss)
        return gradients

class Agentv2(Agentv1):
    def _get_dimensions(self):
        return 3

    def guess(self, o, h):
        """
        x10 times faster than roll
        """
        return np.concatenate([h[1:], o[None]])

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
            epoch (int): Epoch that determine the new subgroup in h5
        """
        data_dict = dict()
        if len(self.training_data) > 0:
            s_t_list, a_t_list, r_t_list, s__t_list, done_t_list = list(zip(*self.training_data))
            data_dict.update({
                '%s/t/s': s_t_list,
                '%s/t/a': a_t_list,
                '%s/t/r': r_t_list,
                '%s/t/s_': s__t_list,
                '%s/t/d': done_t_list,
            })
        if len(self.loss_list) > 0:
            data_dict.update({
                '%s/l': self.loss_list,
            })
        if len(self.eval_data) > 0:
            s_e_list, a_e_list, r_e_list, s__e_list, done_e_list = list(zip(*self.eval_data))
            data_dict.update({
                '%s/e/s': s_e_list,
                '%s/e/a': a_e_list,
                '%s/e/r': r_e_list,
                '%s/e/s_': s__e_list,
                '%s/e/d': done_e_list,
            })
        mode = "a" if epoch > 0 else "w"
        with h5py.File(self.output_data, "a") as f:
            for k, v in data_dict.items():
                try:
                    f[k%str(epoch)] = v
                except (OSError, RuntimeError):
                    del f[k%str(epoch)]
                    f[k%epoch] = v

    def _save_model(self):
        """Save the model of the agent in self.output_model"""
        try:
            self.agent.model.nn.save_weights(self.output_model, save_format='h5')
        except OSError:
            os.makedirs(os.path.dirname(self.output_model))
            self.agent.model.nn.save_weights(self.output_model, save_format='h5')
            

    def _clear_data(self):
        """Clean all data stored from the epoch for the next train step"""
        self.training_data = list()
        self.eval_data = list()
        self.loss_list = list()

    def evalue(self, steps_tot, times, verbose=False, oneline=True, dry_run=False, h_I=None):
        t_ev_s = time.time()
        env = self.env_eval
        policy = Policy(self.agent, h_I)
        trayectory = evalue_policy(policy, env, h_I)
        steps = len(trayectory)
        total_reward = sum([step[2] for step in trayectory])
        action_list = [step[1] for step in trayectory]
        if not dry_run:
            self.eval_data = trayectory
        # action_list = list()
        # while not done:
        #     h = agent.guess(obs, h)
        #     action = agent(h)
        #     obs, reward, done, _ = env.step(action)
        #     action_list.append(action)
        #     if not dry_run:
        #         self.eval_data.append((h, action, reward, obs, done))
        #     total_reward += reward
        #     steps += 1
        # if not dry_run:
            if self.best_reward is None or total_reward > self.best_reward:
                self.best_reward = total_reward
                self._save_model()
                self.early_stop_iterations = 0
            else:
                self.early_stop_iterations += 1
        times['t_ev_tot'] += time.time() - t_ev_s
        if verbose:
            if not dry_run:
                if oneline:
                    template = "W{:.1f}T{:.1f}E{:.1f}| Epoch {:5d} ({:7d} steps) | early {:4d} | exploration {:.8f} | reward {:.8f} | loss {:.8f}"
                    line = template.format(
                                time.time()-times['t_ini'], times['t_tr_tot'], times['t_ev_tot'],
                                self.epoch, steps_tot, self.early_stop_max_iterations - self.early_stop_iterations,
                                self.agent.explore(),
                                total_reward, np.array(self.loss_list).mean()
                            )
                    if self.early_stop_iterations == 0:
                        reinit()
                        # template = "\033[31m" + template + "\033[39m"
                        print(Fore.GREEN + line + Style.RESET_ALL)
                        deinit()
                        solution_array = np.array(action_list).ravel().tolist()
                        if len(solution_array) <= 180
                            print("Solution:", solution_array)
                    else:
                        print(line)
                else:
                    print("Training summary of epoch {} ({} steps):".format(self.epoch, steps))
                    print("  Training mean (std) loss: {} ({})".format(np.array(self.loss_list).mean(), np.array(self.loss_list).std()))
            if not oneline: print("  Evaluation reward: {}".format(total_reward))
            if not dry_run and not oneline:
                print("  Early iter remaining: {}".format(self.early_stop_max_iterations - self.early_stop_iterations))
        if not dry_run:
            self._save_epoch(self.epoch)
            self._clear_data()
            self.epoch += 1

    @property
    def stop(self):
        return self.early_stop_iterations == self.early_stop_max_iterations # == to be able to: early == -1 -> never ends
        
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

class Policy():
    def __init__(self, agent, initial_belief=None):
        self.agent = agent
        if initial_belief is None:
            initial_belief = np.zeros(agent.obs_shape, dtype=np.float32)
        self.internal_belief = initial_belief
    def __call__(self, observation):
        self.internal_belief = self.agent.guess(observation, self.internal_belief)
        return self.agent(self.internal_belief)

StepClass = namedtuple("Step", ["state", "action", "reward", "next_state", "done"])

def evalue_policy(policy, env, h_I=None):
    trayectory = list() # of steps
    next_state = env.reset(h_I)
    done = False
    while not done:
        state = next_state
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        step = StepClass(state, action, reward, next_state, done)
        trayectory.append(step)
    return trayectory

