import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import time

from a2c import A2C
from a2c_baseline import A2CBaseline

def prediction(episodes, agent, duration):
    training_steps = 0
    max_reward = 0
    episode_rewards = []
    episode_steps = []
    action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for episode in range(1, episodes + 1):
        episode_start = time.time()
        print("Episode: ", episode)
        reward, steps, predicted_action_distribution = agent.model_predict()
        for key, _ in action_distribution.items():
            action_distribution[key] += predicted_action_distribution[key]
        training_steps += steps
        episode_rewards.append(reward)
        episode_steps.append(steps)
        max_reward = max(max_reward, reward)
        print("Reward:", reward, "| Steps:", steps)
        episode_end = time.time()
        if (episode_end - episode_start > duration):
            print("Time limit reached")
        break
    return training_steps, max_reward, episode_rewards, episode_steps, action_distribution
