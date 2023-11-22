import numpy as np
import gymnasium as gym
import time
import getopt, sys

from ddqn import DDQN
from a2c import A2C
from a2c_baseline import A2CBaseline
from plotting import Plotting



def main(argv):
    # Hyperparameters
    num_episodes = 500
    steps = 1000
    baseline_steps = 7500
    neurons = 256
    lr = 0.0001
    gamma = 0.95
    duration = 300

    # Environment configuration parameters
    env = gym.make('highway-v0', render_mode='human')
    env.config["lanes_count"] = 3
    env.config["duration"] = duration
    env.config["lane_change_reward"] = 0
    env.config["right_lane_reward"] = 0.2
    env.config["collision_reward"] = -10
    env.config["high_speed_reward"] = 0.8
    env.config["reward_speed_range"] = [30, 40]
    env.config["vehicles_count"] = 60

    try:
        opts, _ = getopt.getopt(argv, "he:s:n:l:g:d:b:", ["help=", "episodes=", "steps=", "neurons=", 
                                                        "learning rate=", "gamma=", "duration=", "baseline steps="])
    except getopt.GetoptError:
        print('Usage: main.py [-h <help>] [-e <episodes>] [-s <steps>] [-n <neurons>] [-l <learning rate>] [-g <gamma>] [-d <duration>] [-n <neurons>] [-b <baseline steps>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: main.py [-e <episodes>] [-s <steps>] [-n <neurons>] [-l <lr>] [-g <gamma>] [-d <duration>] [-b <baseline steps>]')
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
        elif opt in ("-b", "--baselinesteps"):
            baseline_steps = int(arg)

    actor_critic_agent = A2C(env=env, steps=steps, hidden_neurons=neurons, lr=lr, gamma=gamma)
    actor_critic_baseline = A2CBaseline(env=env, steps=baseline_steps, lr=lr, gamma=gamma)

    print("Baseline Training")
    actor_critic_baseline.model_learn()

    baseline_max_reward = 0
    baseline_episode_rewards = []
    baseline_episode_steps = []

    print("Baseline Prediction")
    baseline_total_steps = 0
    baseline_start = time.time()
    baseline_action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for episode in range(1, num_episodes + 1):
        episode_start = time.time()
        print("Episode: ", episode)
        reward, steps, action_distribution = actor_critic_baseline.train_episode()
        for key, _ in baseline_action_distribution.items():
            baseline_action_distribution[key] += action_distribution[key]
        baseline_total_steps += steps
        baseline_episode_rewards.append(reward)
        baseline_episode_steps.append(steps)
        baseline_max_reward = max(baseline_max_reward, reward)
        print("Reward:", reward, "| Steps:", steps)
        episode_end = time.time()
        if (episode_end - episode_start > duration):
            print("Time limit reached")
            break
    baseline_end = time.time()
    baseline_time = baseline_end - baseline_start
    
    max_reward = 0
    episode_rewards = []
    episode_steps = []

    print("A2C Prediction")
    total_steps = 0
    a2c_start = time.time()
    a2c_action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for episode in range(1, num_episodes + 1):
        episode_start = time.time()
        print("Episode: ", episode)
        reward, steps, action_distribution = actor_critic_agent.train_episode()
        for key, _ in a2c_action_distribution.items():
            a2c_action_distribution[key] += action_distribution[key]
        total_steps += steps
        episode_rewards.append(reward)
        episode_steps.append(steps)
        max_reward = max(max_reward, reward)
        print("Reward:", reward, "| Steps:", steps)
        episode_end = time.time()
        if (episode_end - episode_start > duration):
            print("Time limit reached")
            break
    a2c_end = time.time()
    a2c_time = a2c_end - a2c_start

    # Convergence speed
    print("Baseline prediction finished in:", baseline_time, "seconds")
    print("A2C finished in:", a2c_time, "seconds")

    print("Baseline total steps:", baseline_total_steps)
    print("A2C total steps:", total_steps)

    # Max reward
    print("Max baseline reward achieved:", baseline_max_reward)
    print("Max A2C reward achieved:", max_reward)

    plotter = Plotting()
    # Average episode rewards
    plotter.average_episodic_plot(baseline_episode_rewards, episode_rewards, "Reward")

    # Average episode rewards
    plotter.episodic_plot(baseline_episode_rewards, episode_rewards, "Reward")

    # Episode length or number of steps taken
    plotter.episodic_plot(baseline_episode_steps, episode_steps, "Steps")

    # Action distribution
    plotter.bar_graph(baseline_action_distribution, a2c_action_distribution)





if __name__ == "__main__":
    main(sys.argv[1:])