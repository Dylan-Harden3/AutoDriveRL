import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

from agents.a2c import A2C
from baselines.a2c_baseline import A2CBaseline

def prediction(episodes, agent, duration, model):
    episode_rewards = []
    for episode in range(1, episodes + 1):
        print("Episode: ", episode)
        reward, steps = agent.model_predict(model)
        episode_rewards.append(reward)
        print("Reward:", reward, "| Steps:", steps)
    return np.mean(episode_rewards)
