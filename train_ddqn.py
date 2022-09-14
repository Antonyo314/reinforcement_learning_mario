"""Double Deep Q-Network"""

import argparse
import random

import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from ddqn_agent import Agent
from wrappers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--level', type=str, default='SuperMarioBros-1-1-v0')
    parser.add_argument('--checkpoint', type=str)

    args = parser.parse_args()
    render = args.render
    level = args.level
    checkpoint = args.checkpoint

    env = gym_super_mario_bros.make(level)
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    # seed everything
    # TODO looks like  epsilon greedy and env.random() needs additional seeds
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_save_period = 10
    log_period_ = 1
    agent = Agent(action_space_dim=env.action_space.n, level=level, device=device)

    if checkpoint is None:
        print('Training from scratch')
    else:
        agent.load_checkpoint(checkpoint)
        print(f'Training from checkpoints: {checkpoint}')

    episode = 0
    while True:
        state = env.reset()
        while True:
            action = agent.predict_action(state)
            if render:
                env.render()
            next_state, reward, done, info = env.step(action)
            agent.remember(state, next_state, action, reward, done)
            agent.experience_replay(reward)
            state = next_state

            if done or info['flag_get']:
                episode += 1
                agent.log_episode()
                if episode % checkpoint_save_period == 0:
                    agent.save_checkpoint(episode=episode)
                    pass
                if episode % log_period_ == 0:
                    agent.log_period(episode=episode, epsilon=agent.exploration_rate, device=device,
                                     log_period_=log_period_)
                break
