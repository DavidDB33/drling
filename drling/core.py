import copy
import datetime
import logging
logger = logging.getLogger(__file__)
import os
import os.path
import statistics as stt
import sys
import time
from collections import namedtuple
from itertools import chain
from pprint import pprint

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
import tensorflow as tf
from tensorflow.keras import Model, layers as Layers
from tensorflow.keras.layers import Dense, Flatten, Conv1D
# import tensorflow_addons as tfa

class NNv0(Model):
    def __init__(self, n_output, hidden_layers, **kwargs):
        super().__init__(**kwargs)
        self.nnlayers = list()
        for layer in hidden_layers:
            self.nnlayers.append(getattr(Layers, layer['class'])(*layer['args'], **layer['kwargs']))
        self.output_layer = Dense(n_output, activation=None)

    @tf.function(experimental_compile=True)
    def call(self, x):
        y = x
        for layer in self.nnlayers:
            y = layer(y)
        output = self.output_layer(y)
        return output

class NNv1(Model):
    def __init__(self, n_output, **kwargs):
        super().__init__(**kwargs)
        self.d1 = Dense(1024, activation='relu', kernel_initializer="he_uniform")
        self.d2 = Dense(256, activation='relu', kernel_initializer="he_uniform")
        self.d3 = Dense(64, activation='relu', kernel_initializer="he_uniform")
        self.d4 = Dense(16, activation='relu', kernel_initializer="he_uniform")
        self.d5 = Dense(n_output, activation=None)

    @tf.function(experimental_compile=True)
    def call(self, x):
        y = self.d1(x)
        y = self.d2(y)
        y = self.d3(y)
        y = self.d4(y)
        y = self.d5(y)
        return y

class NNv2(Model):
    def __init__(self, n_output, **kwargs):
        super().__init__(**kwargs)
        self.c1 = Conv1D(16, 3, padding='valid', activation='relu', kernel_initializer="glorot_uniform", name="conv1d_1")
        self.c2 = Conv1D(32, 3, padding='valid', activation='relu', kernel_initializer="glorot_uniform", name="conv1d_2")
        self.f1 = Flatten(name="flatten_1")
        self.d1 = Dense(64, activation='relu', kernel_initializer="he_uniform", name="dense_1")
        self.d2 = Dense(16, activation='relu', kernel_initializer="he_uniform", name="dense_2")
        self.d3 = Dense(n_output, activation=None, kernel_initializer="he_uniform", name="dense_3")

    @tf.function(experimental_compile=True)
    def call(self, x):
        y = self.c1(x)
        y = self.f1(y)
        y = self.d1(y)
        y = self.d2(y)
        y = self.d3(y)
        return y

class NNv3(Model):
    def __init__(self, n_output, **kwargs):
        super().__init__(**kwargs)
        self.c1 = Conv1D(32, 3, padding='same', activation='relu', kernel_initializer="glorot_uniform", input_shape=(3, 3), data_format="channels_last", name="conv1d_1")
        self.c2 = Conv1D(64, 3, padding='same', activation='relu', kernel_initializer="glorot_uniform", input_shape=(3, 32), data_format="channels_last", name="conv1d_2")
        self.f1 = Flatten(name="flatten_1")
        self.d1 = Dense(128, activation='relu', kernel_initializer="he_uniform", name="dense_1")
        self.d2 = Dense(64, activation='relu', kernel_initializer="he_uniform", name="dense_2")
        self.d3 = Dense(64, activation='relu', kernel_initializer="he_uniform", name="dense_3")
        self.d4 = Dense(n_output, activation=None, kernel_initializer="he_uniform", name="dense_4")

    @tf.function(experimental_compile=True)
    def call(self, x):
        y = self.c1(x)
        y = self.c2(y)
        y = self.f1(y)
        y = self.d1(y)
        y = self.d2(y)
        y = self.d3(y)
        y = self.d4(y)
        return y

class DQN():
    def __init__(self, observation_space, action_space, name, config, verbose=True):
        self.action_space = action_space
        self.window_size = (config['agent']['history_window'] if 'history_window' in config['agent'] and config['agent']['history_window'] is not None else 1,)
        self.obs_shape = self.window_size + observation_space.shape
        self.n_output = action_space.shape and np.product(action_space.nvec) or action_space.n
        self.loss_object = tf.keras.losses.Huber()# MeanSquaredError() # Try huber loss
        self.learning_rate_schedule = self._get_learning_rate_schedule(config=config)
        # self.optimizer = tf.keras.optimizers.Nadam(learning_rate=config['agent']['network']['learning_rate'], clipnorm=1.0)
        # self.optimizer = tfa.optimizers.AdamW(learning_rate=config['agent']['network']['learning_rate'], weight_decay=self._get_wd(), clipvalue=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule, clipvalue=1.0)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.nn = self._get_nn(self.n_output, config=config, name=name)
        self.nn_target = self._get_nn(self.n_output, config=config, name=name)
        self.gamma = config['agent']['network']['gamma']
        self._build((None, *self.obs_shape), verbose=verbose)

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

    def _build(self, shape_nn, verbose=True):
        self.nn.build(shape_nn)
        if not self.nn_target.built:
            self.nn_target.build(shape_nn)
        if verbose:
            self.nn.summary()

    def _get_nn(self, n_output, config, name):
        H = config['agent']['history_window']
        if 'model' in config:
            nn = NNv0(n_output, hidden_layers=config['model']['hidden_layers'], name=name) # Custom
        elif H > 2:
            nn = NNv3(n_output, name=name) # With conv but less params
        elif H == 1 or H is None:
            nn = NNv1(n_output, name=name) # Without conv
        else:
            raise Exception("Neural Network incompatible")
        if __debug__:
            logger.debug("NN: {}".format(nn))
        return nn

    def load_weights(self, *args, **kwargs):
        ret = self.nn.load_weights(*args, **kwargs)
        self.nn_target.set_weights(self.nn.get_weights())
        return ret

    def save_weights(self, *args, **kwargs):
        return self.nn.save_weights(*args, **kwargs)

    @tf.function(experimental_compile=True)
    def __call__(self, x):
        return self.nn(x)

    @tf.function(experimental_compile=True)
    def qvalue(self, x, a):
        x = self.nn(x)
        a = tf.one_hot(a, self.n_output)
        x = x * a
        x = tf.reduce_sum(x, axis=1)
        return x

    @tf.function(experimental_compile=True)
    def qvalue_with_mask(self, x, mask):
        x = self.nn(x)
        x = tf.multiply(x, mask)
        x = tf.reduce_sum(x, axis=1)
        return x

    @tf.function(experimental_compile=True)
    def qvalues(self, x):
        return self.nn(x)

    @tf.function(experimental_compile=True)
    def qvalue_max(self, x):
        x = self.nn(x)
        x = tf.reduce_max(x, axis=1)
        return x

    @tf.function(experimental_compile=True)
    def target_qvalue_max(self, x):
        x = self.nn_target(x)
        x = tf.reduce_max(x, axis=1)
        return x

    @tf.function(experimental_compile=True)
    def argmax_qvalue(self, x):
        x = self.nn(x)
        x = tf.argmax(x, axis=1)
        return x

    @tf.function(experimental_compile=True)
    def qtarget(self, x, r, d):
        return r + self.gamma*self.target_qvalue_max(x)*(1-d)

    @tf.function(experimental_compile=True)
    def compute_error(self, o, a, r, n_o, done):
        return tf.keras.losses.MSE(self.qtarget(n_o, r), self.qvalue(o, a))

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
        self.best_loss = float("inf")
        self.ndim = self._get_dimensions()
        self.memory = memory
        self.target_steps_update = int(config['agent']['target']['update'])
        self.train_steps_without_update = 0
        self.train_steps_to_update = self.target_steps_update
        self.percentage_to_update_init = 1.1
        self.percentage_to_update = self.percentage_to_update_init
        self.percentage_to_update_target = 1
        self.actual_loss = 1.0
        self.update_msg = "" # For debugging purposes
        self.update_flag = False # For debugging purposes

    def __call__(self, *args, **kwargs):
        """Alias for self.act"""
        return self.act(*args, **kwargs)

    def _get_dimensions(self):
        return 2

    @property
    def obs_shape(self):
        return self.model.obs_shape

    def guess_init(self):
        return np.zeros(self.obs_shape, dtype=np.float32)

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
            if not keep_tensor:
                act = act.numpy()
        else:
            act = self.model.argmax_qvalue(belief)
            if not keep_tensor:
                act = int(act)
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
    
    def _convert_experience_list_to_tensor(self, experience_list):
        hstate_list, action_list, reward_list, next_obs_list, done_list = list(zip(*experience_list))
        next_hstate_list = [self.guess(obs, hstate) for hstate, obs in zip(hstate_list, next_obs_list)]
        hstate_tensor = tf.convert_to_tensor(hstate_list, dtype=tf.float32)
        action_tensor = tf.squeeze(tf.convert_to_tensor(action_list, dtype=tf.int32))
        reward_tensor = tf.convert_to_tensor(reward_list, dtype=tf.float32)
        next_hstate_tensor = tf.convert_to_tensor(next_hstate_list, dtype=tf.float32)
        done_tensor = tf.convert_to_tensor(done_list, dtype=tf.float32)
        return hstate_tensor, action_tensor, reward_tensor, next_hstate_tensor, done_tensor

    def train_step(self):
        experience_list = self.memory.sample(self.batch_size)
        hstate_tensor, action_tensor, reward_tensor, next_hstate_tensor, done_tensor = self._convert_experience_list_to_tensor(experience_list)
        self.tf_train_step(hstate_tensor, action_tensor, reward_tensor, next_hstate_tensor, done_tensor)
        loss = self.model.train_loss.result().numpy()
        self.actual_loss = self.actual_loss + 0.1*(loss - self.actual_loss) # Moving average to smooth the update
        self.model.train_loss.reset_states()
        if self.train_steps_without_update >= self.train_steps_to_update and self.actual_loss < self.percentage_to_update*self.best_loss:
            self.model.nn_target.set_weights(self.model.nn.get_weights())
            self.update_msg = "Policy updated after {} steps".format(self.train_steps_without_update)
            self.update_flag = True
            self.train_steps_without_update = 0
            # if self.train_steps_to_update < self.target_steps_update:
            #     self.train_steps_to_update += 1
            if self.actual_loss < self.best_loss:
                self.best_loss = min(self.actual_loss, self.best_loss)
                self.percentage_to_update = self.percentage_to_update_init
            else: # Contract update restriction
                self.percentage_to_update = self.percentage_to_update_target + 0.9*(self.percentage_to_update - self.percentage_to_update_target)
        else:
            self.train_steps_without_update += 1
        return loss

    @tf.function(experimental_compile=True)
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
        x10 times faster than np.roll
        """
        return np.concatenate([h[1:], o[None]])

class Monitor():
    def __init__(self, agent, env_eval_list, times=None, config=None, output_template=None):
        self.config = config
        self.agent = agent
        self.env_eval_list = env_eval_list
        print("{:=^65}".format("ENVIRONMENT INFO"))
        for env in self.env_eval_list:
            pprint(env.spec.__dict__)
            print(f"Start step values: {np.array(env.step_start_values)}")
        print("{:=^65}".format("END ENVIRONMENT INFO"))
        self.time_last_msg = time.time()
        self.filename_data = output_template.format(object='data')
        self.filename_model = output_template.format(object='model')
        self.filename_early = output_template.format(object='early')
        self.early_stop_iterations = 0
        self.early_stop_max_iterations = config['agent']['monitor']['early_stop']
        self.best_reward = None
        self.plot_epoch = 0
        self.plot_rewards = list()
        self.plot_displayed = None
        self.training_trajectory_list = list()
        self.loss_list = list()
        self.times = times
        self.ema = 0
        self.GAP = 0.01 # 1% gap from optimal to mark as solved
        self._stop = False

    def _evalue(self, env, h_I=None):
        t_ev_s = time.time()
        policy = Policy(self.agent, h_I)
        trajectory = evalue_policy(policy, env, h_I)
        steps = len(trajectory)
        action_list = [step.action for step in trajectory]
        self.times['t_ev_tot'] += time.time() - t_ev_s
        return trajectory

    def _verbose(self, epoch, step, loss_list, cumulated_reward_list, eval_trajectory_list, oneline=True, dry_run=False):
        # TODO: Check flow code:
        #       Often values to be printed are calculated "a priori" but then conditions to be printed aren't met being useless the first calculations
        def compute_gap(value, threshold):
            if threshold is None:
                return float("nan")
            return (value - threshold) / threshold
        reward_threshold_list = [env.spec.reward_threshold for env in self.env_eval_list]
        loss_array = np.array(loss_list)
        loss_mean, loss_std = (loss_array.mean(), loss_array.std()) if loss_array.size > 0 else (np.nan, np.nan)
        reward_gap_list = list(map(compute_gap, cumulated_reward_list, reward_threshold_list))
        if oneline:
            # template = "W{:.1f}T{:.1f}E{:.1f} | Epoch {:5d} ({:7d} steps) | early {:4d} | exploration {:.8f} | reward {:.8f} | loss mean {:.8f} | loss std {:.8f} {}"
            delta_time = time.time()-self.times['t_ini']
            # steps_per_second = int(step//delta_time)
            self.ema = compute_ema_from_list(self.ema, self.times['t_step_delta_list']) # Exponential moving average
            self.times['t_step_delta_list'] = list()
            template = ("{:.1f}s | Ech {:3d} (step {:5d} {:4.1f}/s) | ear {:4d} | ε {:.4f} | lmean {:7.4e} | lstd {:7.4e} |"
                + " |".join([" r {:.4f} (gap {:.2f})"]*len(cumulated_reward_list)) + " {}")
            line = template.format(
                        delta_time, # times['t_tr_tot'], times['t_ev_tot'],
                        epoch, step, self.ema, self.early_stop_max_iterations - self.early_stop_iterations,
                        self.agent.explore(),
                        loss_mean, loss_std,
                        *chain.from_iterable([(cumulated_reward, reward_gap) for cumulated_reward, reward_gap in zip(cumulated_reward_list, reward_gap_list)]),
                        self.agent.update_msg,
                    )
            if self.early_stop_iterations == 0:
                if os.isatty(sys.stdout.fileno()):
                    # template = "\033[31m" + template + "\033[39m"
                    reinit()
                    print(Fore.GREEN + line + Style.RESET_ALL, flush=True)
                    deinit()
                else:
                    print("> " + line, flush=True)
                solution_array = np.array([step.action for step in eval_trajectory_list[0]]).ravel().tolist()
                if len(solution_array) <= 40:
                    print("Solution:", solution_array)
            elif time.time() - self.time_last_msg > 1.0 or self.agent.update_flag:
                print(line, flush=True)
                self.time_last_msg = time.time()
            if self.agent.update_flag: # If update_flag is True then update_msg has text. After print, clean
                self.agent.update_msg = ""
                self.agent.update_flag = False # To avoid reassing every time (cheaper check)
        else:
            print("Training summary of epoch {} ({} steps):".format(epoch, steps))
            print("  Training mean (std) loss: {} ({})".format(loss_mean, loss_std))
            print("  Evaluation reward: {}".format(cumulated_reward_list))
            print("  Early iter remaining: {}".format(self.early_stop_max_iterations - self.early_stop_iterations))
        # self.plot_reward(cumulated_reward_list[-1])

    def _save_model(self, filename_model):
        """Save the model of the agent in filename_model"""
        try:
            self.agent.model.nn.save_weights(filename_model, save_format='h5')
        except OSError:
            os.makedirs(os.path.dirname(filename_model))
            self.agent.model.nn.save_weights(filename_model, save_format='h5')

    @staticmethod
    def _dump_loss(h5key, loss_list):
        """Get a dict of the loss to be stored in h5
        Args:
            loss_list (list<float>): All the loss from one epoch
            epoch (int): Epoch that determine the new subgroup in h5
        """
        data_dict = dict()
        if len(loss_list) > 0:
            data_dict = {
                h5key: loss_list,
            }
        return data_dict

    @staticmethod
    def _dump_trajectory(template, trajectory):
        """Get h5 trajectory format
        Args:
            trajectory (list<StepClass>): A trajectory of one epoch
        Returns:
            data_dict (dict): A dict to be used in h5
        """
        data_dict = dict()
        if len(trajectory) > 0:
            s_t_list, a_t_list, r_t_list, s__t_list, done_t_list = list(zip(*trajectory))
            data_dict = {
                template.format(target='s'): s_t_list,
                template.format(target='a'): a_t_list,
                template.format(target='r'): r_t_list,
                template.format(target='s_'): s__t_list,
                template.format(target='d'): done_t_list,
            }
        return data_dict

    @staticmethod
    def _save_data(filename_data, data_dict, mode="a"):
        with h5py.File(filename_data, mode) as f:
            for k, v in data_dict.items():
                try:
                    f[k] = v
                except (OSError, RuntimeError) as e:
                    del f[k]
                    f[k] = v

    def _save(self, epoch, step, eval_trajectory_list, clean_data=True):
        template = "{{phase}}/{epoch}/{step}/{{target}}".format(epoch=epoch, step=step)
        self._save_model(self.filename_model)
        if self.early_stop_iterations == 0:
            self._save_model(self.filename_early)
        data_dict = dict()
        data_dict.update(self._dump_loss(template.format(phase="t", target="l"), self.loss_list))
        data_dict.update(self._dump_trajectory(template.format(phase="t", target="{target}"), self.training_trajectory_list))
        for i, trajectory in enumerate(eval_trajectory_list):
            # print(sum([s.reward for s in trajectory]))
            data_dict.update(self._dump_trajectory(template.format(phase="e-%d"%i, target="{target}"), trajectory))
        mode = "w" if step == 0 else "a"
        self._save_data(self.filename_data, data_dict, mode=mode)
        if clean_data:
            self.training_trajectory_list = list()
            self.loss_list = list()

    def evalue(self, step, epoch, verbose=False, oneline=True, dry_run=False, h_I=None):
        # eval_trajectory_list = {"t": self.training_trajectory_list}
        eval_trajectory_list = list()
        cumulated_reward_list = list()
        for i, env in enumerate(self.env_eval_list):
            trajectory = self._evalue(env, h_I)
            cumulated_reward = sum([step.reward for step in trajectory])
            cumulated_reward_list.append(cumulated_reward)
            eval_trajectory_list.append(trajectory)
            if (i == 0
                    and env.spec.reward_threshold is not None
                    and cumulated_reward >= env.spec.reward_threshold * (1 + self.GAP) # GAP = (cumulated_reward - reward_threshold) / reward_threshold
                    ): 
                self._stop = True
        if self.best_reward is None or cumulated_reward_list[-1] > self.best_reward:
            self.best_reward = cumulated_reward_list[-1]
            self.early_stop_iterations = 0
        else:
            self.early_stop_iterations += 1
        if verbose:
            self._verbose(epoch, step, self.loss_list, cumulated_reward_list, eval_trajectory_list, oneline=oneline, dry_run=dry_run)
        if not dry_run:
            self._save(epoch, step, eval_trajectory_list) # Save early model and trajectory data

    def add_loss(self, loss):
        self.loss_list.append(loss)

    def add_experience(self, s, a, r, s_, done):
        self.training_trajectory_list.append(StepClass(s, a, r, s_, done))

    @property
    def stop(self):
        return self._stop or self.early_stop_iterations == self.early_stop_max_iterations # == to be able to: early == -1 -> never ends
        
    @property
    def has_improved(self):
        raise NotImplementedError("BLA")
        return self._has_improved

    def debug(self, agent):
        print("\tDEBUG: Exploration: %f"%agent.explore())

    def plot_reward(self, reward):
        self.plot_rewards.append(reward)
        if self.plot_epoch == 0:
            plt.ion()
            plt.clf()
            self.last_reward = reward
            plt.show()
        plt.plot([self.plot_epoch, self.plot_epoch+1], [self.last_reward, reward], color='b')
        plt.pause(0.00001)
        self.plot_epoch += 1
        self.last_reward = reward

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
    trajectory = list() # of steps
    next_state = env.reset(obs=h_I)
    done = False
    while not done:
        state = next_state
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        step = StepClass(state, action, reward, next_state, done)
        trajectory.append(step)
    return trajectory

def compute_ema(ema0, y_value, alpha=0.1):
    """ema_t+1 = (1-α)ema + αy"""
    assert isinstance(y_value, (int, float)), "y_value has to be a number"
    return (1-alpha)*ema0 + alpha*(y_value)

def compute_ema_from_list(ema0, y_values, alpha=0.1):
    """ema_t+n = (1-α)^n*ema + αΣ(1-α)^(n-i)*y_i"""
    assert isinstance(y_values, list), "y_values has to be a list"
    n = len(y_values)
    alphy = 1-alpha
    return (alphy)**n*ema0 + alpha*sum([y_val*alphy**(n-1-i) for i,y_val in enumerate(y_values)])
