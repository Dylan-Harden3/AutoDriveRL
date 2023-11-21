import numpy as np
import gymnasium as gym
import time

from ddqn import DDQN
from a2c import A2C
from matplotlib import pyplot as plt



def main():
    env = gym.make('highway-v0', render_mode='human')

    # Hyperparameters
    num_episodes = 3000
    steps = 1000
    neurons = 128
    lr = 0.001
    gamma = 0.99

    actor_critic_agent = A2C(env=env, steps=steps, hidden_neurons=neurons, lr=lr, gamma=gamma)
    
    max_reward = 0
    episode_rewards = []

    start = time.time()
    for episode in range(1, num_episodes + 1):
        print("Episode: ", episode)
        reward, steps = actor_critic_agent.train_episode()
        episode_rewards.append(reward)
        max_reward = max(max_reward, reward)
        print("Reward:", reward, "| Steps:", steps)
    end = time.time()
    print("Finished in:", end - start, "seconds")
    print("Max reward achieved:", max_reward)

    average_rewards = [sum(episode_rewards[:i+1]) / len(episode_rewards[:i+1]) for i in range(len(episode_rewards))]
    plt.plot(average_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.show()
    


if __name__ == "__main__":
    main()