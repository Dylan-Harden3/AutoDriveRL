import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import time

from agents.a2c import A2C
from baselines.a2c_baseline import A2CBaseline

def prediction(episodes, agent, duration, model):
    training_steps = 0
    max_reward = 0
    episode_rewards = []
    episode_steps = []
    action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for episode in range(1, episodes + 1):
        print("Episode: ", episode)
        reward, steps, predicted_action_distribution = agent.model_predict(model)
        for key, _ in action_distribution.items():
            action_distribution[key] += predicted_action_distribution[key]
        training_steps += steps
        episode_rewards.append(reward)
        episode_steps.append(steps)
        max_reward = max(max_reward, reward)
        print("Reward:", reward, "| Steps:", steps)
    return training_steps, max_reward, episode_rewards, episode_steps, action_distribution
