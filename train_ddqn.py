"""Double Deep Q-Network"""

import random

import gym_super_mario_bros
import numpy as np
import torch
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from wrappers import *

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])

# seed everything
env.seed(0)
env.action_space.seed(0)
torch.manual_seed(0)
torch.random.manual_seed(0)
random.seed(0)
np.random.seed(0)

# apply wrappers
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)


class Agent:
    pass