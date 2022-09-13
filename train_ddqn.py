"""Double Deep Q-Network"""

import argparse
import copy
import datetime
import os
import random
from collections import deque

import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from torch import nn

from wrappers import *

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
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


class MarioNet(nn.Module):
    def __init__(self, state_dim, output_dim):
        super().__init__()
        in_channels, _, _ = state_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # froze target_network
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)


class Agent:
    def __init__(self, action_space_dim):
        self.action_space_dim = action_space_dim
        if device == 'cuda':
            self.net = MarioNet(state_dim=(4, 84, 84), output_dim=self.action_space_dim).cuda()
        else:
            self.net = MarioNet(state_dim=(4, 84, 84), output_dim=self.action_space_dim)

        self.save_dir = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99
        self.exploration_rate_min = 0.01

        self.current_step = 0
        self.memory = deque(maxlen=100_000)
        self.batch_size = 128

        self.gamma = 0.95  # discount factor
        self.sync_period = 10_000

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025, eps=1e-4)

        self.loss = torch.nn.SmoothL1Loss()
        self.episode_rewards = []
        self.moving_average_episode_rewards = []
        self.current_episode_reward = 0

    def remember(self, state, next_state, action, reward, done):
        state = torch.FloatTensor(state.__array__())
        next_state = torch.FloatTensor(next_state.__array__())
        action = torch.LongTensor([action])
        reward = torch.DoubleTensor([reward])
        done = torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def experience_replay(self, step_reward):
        self.current_episode_reward += step_reward
        if self.current_step % self.sync_period == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())
        if self.batch_size > len(self.memory):
            return
        state, next_state, action, reward, done = self.recall()
        if device == 'cuda':
            q_estimate = self.net(state.cuda(), model="online")[np.arange(0, self.batch_size), action.cuda()]
        else:
            q_estimate = self.net(state, model='online')[np.arange(0, self.batch_size), action]

        with torch.no_grad():
            if device == 'cuda':
                best_action = torch.argmax(self.net(next_state.cuda(), model="online"), dim=1)
                next_q = self.net(next_state.cuda(), model="target")[np.arange(0, self.batch_size), best_action]
                q_target = (reward.cuda() + (1 - done.cuda().float()) * self.gamma * next_q).float()
            else:
                best_action = torch.argmax(self.net(next_state, model='online'), dim=1)
                next_q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
                q_target = (reward + (1 - done.float()) * self.gamma * next_q).float()
        loss = self.loss(q_estimate, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict_action(self, state):
        if np.random.rand() < self.exploration_rate:
            # Exploration
            action = np.random.randint(self.action_space_dim)
        else:
            # Exploitation
            if device == 'cuda':
                state = torch.tensor(state.__array__()).cuda().unsqueeze(0)
            else:
                state = torch.tensor(state.__array__()).unsqueeze(0)
            action_values = self.net(state, model='online')
            action = torch.argmax(action_values, dim=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.current_step += 1

        return action

    def load_checkpoint(self, path):
        if device == 'cuda':
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']

    def save_checkpoint(self, episode):
        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(self.save_dir, f'checkpoint_{episode}.pth')
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), f=filename)
        print(f"Checkpoint saved to '{filename}'")

    def log_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0

    def log_period(self, episode, epsilon, device):
        self.moving_average_episode_rewards.append(np.round(np.mean(self.episode_rewards[-log_period_:]), 3))
        print(
            f'Episode {episode: <5} - Epsilon {epsilon: <5} - Mean Reward {self.moving_average_episode_rewards[-1]: <5} - Device {device: <5}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    render = args.render
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_save_period = 10
    log_period_ = 1
    agent = Agent(action_space_dim=env.action_space.n)

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
                    agent.log_period(episode=episode, epsilon=agent.exploration_rate, device=device)
                break
