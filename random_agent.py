from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np


class MarioAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.rewards = list()
        self.gamma = 0.99  # discount factor

    def predict_next_action(self, state):
        np.random.seed(np.sum(state))
        return np.random.randint(self.action_dim)

    def accumulate_rewards(self, reward):
        self.rewards.append(reward)

    def get_total_reward(self):
        return sum(self.rewards)

    def calculate_return(self):
        _return = 0
        for i, reward in enumerate(self.rewards):
            _return += self.gamma ** i * reward
        return _return


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])  # move right or right and up

n_episods = 10
for episode in range(n_episods):  # on every episode Mario has 3 lifes
    state = env.reset()
    env.seed(0)
    agent = MarioAgent(action_dim=env.action_space.n)

    while True:
        env.render()

        action = agent.predict_next_action(state=state)
        next_state, reward, done, info = env.step(
            action)  # reward: https://github.com/Kautenja/gym-super-mario-bros#reward-function

        agent.accumulate_rewards(reward=reward)
        state = next_state
        if done:
            print(f'total_reward: {agent.get_total_reward()}')
            print(f'return: {agent.calculate_return()}')
            print('---')
            break
