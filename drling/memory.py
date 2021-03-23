import random
from collections import deque
import numpy as np
from tqdm import autonotebook as tqdm

class Memoryv1():
    def __init__(self, max_size=100000, min_size=1000, seed=None, verbose=False):
        self.buffer = deque(maxlen = max_size)
        self._rng = random.Random() # The most efficient way for sampling from a big list
        self.min_size = min_size
        self.min_size_iterable = self._get_min_size_iterable(min_size, verbose)
        self.set_seed(seed)

    def set_seed(self, seed=None):
        """Set seed for sampling from the memory buffer"""
        if seed is not None:
            self._rng.seed(seed)

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
        """Add an experience to the buffer
        An experience is often defined as a (s, a, r, s', d) tuple where any object can be of any type
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sampling from the buffer"""
        return self._rng.sample(self.buffer, batch_size)

