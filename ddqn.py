import gymnasium as gym
import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
from lib import plotting


env = gym.make('highway-v0', render_mode='human')

class QFunction(nn.Module):
    """
    Q-network definition.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
    ):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)

class DDQN():
    def train_episode(self):
        state, _ = env.reset()
        done = truncated = False
        for _ in range(self.options.steps):
            action = ... # Your agent code here
            next_state, reward, done, _, _ = env.step(action)

            if done:
                break

            state = next_state