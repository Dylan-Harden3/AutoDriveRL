import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time

from ddqn import DDQN
from a2c import A2C
from torch.optim import Adam



def main():
    env = gym.make('highway-v0', render_mode='human')
    num_episodes = 500

    actor_critic_agent = A2C(env=env, steps=1000, hidden_neurons=128, lr=0.001, gamma=0.99)
    
    for episode in range(1, num_episodes + 1):
        print("Episode: ", episode)
        print("Episode Reward: ", actor_critic_agent.train_episode())

if __name__ == "__main__":
    main()