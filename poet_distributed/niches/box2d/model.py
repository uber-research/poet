# The following code is modified from hardmaru/estool (https://github.com/hardmaru/estool/) under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.


import numpy as np
import random
import json
from .env import make_env
import time
import logging
logger = logging.getLogger(__name__)

final_mode = False
render_mode = False
RENDER_DELAY = False
record_video = False
MEAN_MODE = False


def make_model(game):
    # can be extended in the future.
    model = Model(game)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def passthru(x):
    return x

# useful for discrete actions


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# useful for discrete actions


def sample(p):
    return np.argmax(np.random.multinomial(1, p))


class Model:
    ''' simple feedforward model '''

    def __init__(self, game):
        self.output_noise = game.output_noise
        self.env_name = game.env_name
        self.layer_1 = game.layers[0]
        self.layer_2 = game.layers[1]
        self.rnn_mode = False  # in the future will be useful
        self.time_input = 0  # use extra sinusoid input
        self.sigma_bias = game.noise_bias  # bias in stdev of output
        self.sigma_factor = 0.5  # multiplicative in stdev of output
        if game.time_factor > 0:
            self.time_factor = float(game.time_factor)
            self.time_input = 1
        self.input_size = game.input_size
        self.output_size = game.output_size
        self.shapes = [(self.input_size + self.time_input, self.layer_1),
                       (self.layer_1, self.layer_2),
                       (self.layer_2, self.output_size)]

        self.sample_output = False
        if game.activation == 'relu':
            self.activations = [relu, relu, passthru]
        elif game.activation == 'sigmoid':
            self.activations = [np.tanh, np.tanh, sigmoid]
        elif game.activation == 'softmax':
            self.activations = [np.tanh, np.tanh, softmax]
            self.sample_output = True
        elif game.activation == 'passthru':
            self.activations = [np.tanh, np.tanh, passthru]
        else:
            self.activations = [np.tanh, np.tanh, np.tanh]

        self.weight = []
        self.bias = []
        self.bias_log_std = []
        self.bias_std = []
        self.param_count = 0

        idx = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])
            if self.output_noise[idx]:
                self.param_count += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.bias_log_std.append(log_std)
            out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
            self.bias_std.append(out_std)
            idx += 1

        self.render_mode = False

    def __repr__(self):
        return "{}".format(self.__dict__)

    def make_env(self, seed, render_mode=False, env_config=None):
        self.render_mode = render_mode
        self.env = make_env(self.env_name, seed=seed,
                            render_mode=render_mode, env_config=env_config)

    def get_action(self, x, t=0, mean_mode=False):
        # if mean_mode = True, ignore sampling.
        h = np.array(x).flatten()
        if self.time_input == 1:
            time_signal = float(t) / self.time_factor
            h = np.concatenate([h, [time_signal]])
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            if (self.output_noise[i] and (not mean_mode)):
                out_size = self.shapes[i][1]
                out_std = self.bias_std[i]
                output_noise = np.random.randn(out_size) * out_std
                h += output_noise
            h = self.activations[i](h)

        if self.sample_output:
            h = sample(h)

        return h

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s
            if self.output_noise[i]:
                s = b_shape
                self.bias_log_std[i] = np.array(
                    model_params[pointer:pointer + s])
                self.bias_std[i] = np.exp(
                    self.sigma_factor * self.bias_log_std[i] + self.sigma_bias)
                if self.render_mode:
                    print("bias_std, layer", i, self.bias_std[i])
                pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count) * stdev

def simulate(model, seed, train_mode=False, render_mode=False, num_episode=5,
             max_len=-1, env_config_this_sim=None):
    reward_list = []
    t_list = []

    max_episode_length = 2000

    if train_mode and max_len > 0:
        if max_len < max_episode_length:
            max_episode_length = max_len

    if (seed >= 0):
        logger.debug('Setting seed to {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    if env_config_this_sim:
        model.env.set_env_config(env_config_this_sim)

    for _ in range(num_episode):

        if model.rnn_mode:
            model.reset()

        obs = model.env.reset()
        if obs is None:
            obs = np.zeros(model.input_size)

        total_reward = 0.0
        for t in range(max_episode_length):

            if render_mode:
                model.env.render("human")
                if RENDER_DELAY:
                    time.sleep(0.01)

            if model.rnn_mode:
                model.update(obs, t)
                action = model.get_action()
            else:
                if MEAN_MODE:
                    action = model.get_action(
                        obs, t=t, mean_mode=(not train_mode))
                else:
                    action = model.get_action(obs, t=t, mean_mode=False)

            obs, reward, done, info = model.env.step(action)
            total_reward += reward

            if done:
                break

        if render_mode:
            print("reward", total_reward, "timesteps", t)
        reward_list.append(total_reward)
        t_list.append(t)

    return reward_list, t_list
