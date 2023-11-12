import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam

from lib import plotting

env = gym.make('highway-v0', render_mode='human')

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        # Shared layers
        for i in range(len(sizes) - 2):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Actor head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], act_dim))
        # Critic head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[i](x))
        # Actor head
        probs = F.softmax(self.layers[-2](x), dim=-1)
        # Critic head
        value = self.layers[-1](x)

        return torch.squeeze(probs, -1), torch.squeeze(value, -1)


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
