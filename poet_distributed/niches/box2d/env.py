# The following code is modified from hardmaru/estool (https://github.com/hardmaru/estool/) under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.


from collections import namedtuple
# import gym
from .bipedal_walker_custom import BipedalWalkerCustom, Env_config  # noqa


def make_env(env_name, seed, render_mode=False, env_config=None):
    if env_name.startswith("BipedalWalkerCustom"):
        assert env_config is not None
        env = BipedalWalkerCustom(env_config)
    else:
        # env = gym.make(env_name)
        raise Exception('Got env_name {}'.format(env_name))
    if render_mode and not env_name.startswith("Roboschool"):
        env.render("human")
    if (seed >= 0):
        env.seed(seed)

    # print("environment details")
    # print("env.action_space", env.action_space)
    # print("high, low", env.action_space.high, env.action_space.low)
    # print("environment details")
    # print("env.observation_space", env.observation_space)
    # print("high, low", env.observation_space.high, env.observation_space.low)
    # assert False

    return env


Game = namedtuple('Game', ['env_name', 'time_factor', 'input_size',
                           'output_size', 'layers', 'activation', 'noise_bias',
                           'output_noise'])

bipedhard_custom = Game(env_name='BipedalWalkerCustom-v0',
                        input_size=24,
                        output_size=4,
                        time_factor=0,
                        layers=[30, 30],
                        activation='tanh',
                        noise_bias=0.0,
                        output_noise=[False, False, False],
                        )
