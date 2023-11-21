import numpy as np
import gymnasium as gym
import time

from ddqn import DDQN
from a2c import A2C



def main():
    env = gym.make('highway-v0', render_mode='human')

    # Hyperparameters
    num_episodes = 2000
    steps = 1000
    neurons = 128
    lr = 0.001
    gamma = 0.99

    actor_critic_agent = A2C(env=env, steps=steps, hidden_neurons=neurons, lr=lr, gamma=gamma)
    
    start = time.time()
    for episode in range(1, num_episodes + 1):
        reward, steps = actor_critic_agent.train_episode()
        print("Episode: ", episode)
        print("Reward:", reward, "| Steps:", steps)
    end = time.time()
    print("Finished in:", end - start, "seconds")
    
if __name__ == "__main__":
    main()