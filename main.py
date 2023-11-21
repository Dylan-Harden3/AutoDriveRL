import numpy as np
import gymnasium as gym
import time
import getopt, sys

from ddqn import DDQN
from a2c import A2C
from matplotlib import pyplot as plt



def main(argv):
    # Hyperparameters
    num_episodes = 3000
    steps = 1000
    neurons = 128
    lr = 0.001
    gamma = 0.99
    duration = 120

    env = gym.make('highway-fast-v0', render_mode='human')
    env.config["duration"] = duration
    env.config["lane_change_reward"] = 0.05
    env.config["right_lane_reward"] = 0.2
    env.config["collision_reward"] = -10
    env.config["high_speed_reward"] = 0.8
    env.config["reward_speed_range"] = [30, 40]
    env.config["vehicles_count"] = 60

    try:
        opts, _ = getopt.getopt(argv, "he:s:n:l:g:d:", ["help=", "episodes=", "steps=", "neurons=", "learning rate=", "gamma=", "duration="])
    except getopt.GetoptError:
        print('Usage: main.py [-h <help>] [-e <episodes>] [-s <steps>] [-n <neurons>] [-l <lr>] [-g <gamma>] [-d <duration>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: main.py [-e <episodes>] [-s <steps>] [-n <neurons>] [-l <lr>] [-g <gamma>] [-d <duration>]')
            sys.exit()
        elif opt in ("-e", "--episodes"):
            num_episodes = int(arg)
        elif opt in ("-s", "--steps"):
            steps = int(arg)
        elif opt in ("-n", "--neurons"):
            neurons = int(arg)
        elif opt in ("-l", "--lr"):
            lr = float(arg)
        elif opt in ("-g", "--gamma"):
            gamma = float(arg)
        elif opt in ("-d", "--duration"):
            duration = float(arg)
            env.config["duration"] = duration

    actor_critic_agent = A2C(env=env, steps=steps, hidden_neurons=neurons, lr=lr, gamma=gamma)
    
    max_reward = 0
    episode_rewards = []

    start = time.time()
    for episode in range(1, num_episodes + 1):
        episode_start = time.time()
        print("Episode: ", episode)
        reward, steps = actor_critic_agent.train_episode()
        episode_rewards.append(reward)
        max_reward = max(max_reward, reward)
        print("Reward:", reward, "| Steps:", steps)
        episode_end = time.time()
        if (episode_end - episode_start > duration):
            print("Time limit reached")
            break
    end = time.time()
    print("Finished in:", end - start, "seconds")
    print("Max reward achieved:", max_reward)

    #Plot average rewards 
    average_rewards = [sum(episode_rewards[:i+1]) / len(episode_rewards[:i+1]) for i in range(len(episode_rewards))]
    plt.plot(average_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.show()
    
    #Plot episodic rewards
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward per Episode')
    plt.title('Episode Reward over Time')
    plt.show()



if __name__ == "__main__":
    main(sys.argv[1:])