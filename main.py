import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time

from ddqn import DDQN
from a2c import A2C
from torch.optim import Adam

from lib import plotting


def main():
    env = gym.make('highway-v0', render_mode='human')
    num_episodes = 100

    ddqn_agent = DDQN(env=env)
    actor_critic_agent = A2C(env=env)

    for _ in range(num_episodes):
        ddqn_agent.train_episode() 
        actor_critic_agent.train_episode()  

if __name__ == "__main__":
    main()