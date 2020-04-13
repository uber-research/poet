# The following code is modified from openai/evolution-strategies-starter
# (https://github.com/openai/evolution-strategies-starter)
# under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.


import numpy as np


class Optimizer(object):
    def __init__(self, theta):
        self.dim = len(theta)
        self.t = 0

    def update(self, theta, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return ratio, theta + step

    def _compute_step(self, globalg):
        raise NotImplementedError


class SimpleSGD(Optimizer):
    def __init__(self, stepsize):
        self.stepsize = stepsize

    def compute(self, theta, globalg):
        step = -self.stepsize * globalg
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return ratio, theta + step


class SGD(Optimizer):
    def __init__(self, theta, stepsize, momentum=0.9):
        Optimizer.__init__(self, theta)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, theta, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, theta)
        self.stepsize = stepsize
        self.init_stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def reset(self):
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.t = 0
        self.stepsize = self.init_stepsize

    def _compute_step(self, globalg):
        a = self.stepsize * \
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def propose(self, theta, globalg):
        a = self.stepsize * \
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        m = self.beta1 * self.m + (1 - self.beta1) * globalg
        v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * m / (np.sqrt(v) + self.epsilon)
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return ratio, theta + step
