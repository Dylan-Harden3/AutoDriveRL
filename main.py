import numpy as np
import gymnasium as gym
import time
import getopt, sys
import os

from ddqn import DDQN
from a2c import A2C
from a2c_baseline import A2CBaseline
from plotting import Plotting
from tensorflow.keras import models



def main(argv):
    # Hyperparameters
    training_steps = 7500
    num_episodes = 100 
    testing_steps = 1000
    neurons = 256
    lr = 0.0001
    gamma = 0.95
    duration = 150

    # Environment configuration parameters
    env = gym.make('highway-v0', render_mode='human')
    env.config["lanes_count"] = 5
    env.config["duration"] = duration
    env.config["lane_change_reward"] = 0
    env.config["right_lane_reward"] = 0.2
    env.config["collision_reward"] = -5
    env.config["high_speed_reward"] = 0.8
    env.config["reward_speed_range"] = [30, 40]
    env.config["vehicles_count"] = 60

    try:
        opts, _ = getopt.getopt(argv, "he:s:t:n:l:g:d:", ["help=", "episodes=", "steps=", "testing steps=" "neurons=", 
                                                        "learning rate=", "gamma=", "duration="])
    except getopt.GetoptError:
        print('Usage: main.py [-h <help>] [-e <episodes>] [-s <steps>] [-t <testing steps>] [-n <neurons>] [-l <learning rate>] [-g <gamma>] [-d <duration>] [-n <neurons>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: main.py [-e <episodes>] [-s <steps>] [-t <testing steps>] [-n <neurons>] [-l <lr>] [-g <gamma>] [-d <duration>] [-b <baseline steps>]')
            sys.exit()
        elif opt in ("-e", "--episodes"):
            num_episodes = int(arg)
        elif opt in ("-s", "--training_steps"):
            training_steps = int(arg)
        elif opt in ("-t", "--testing_steps"):
            testing_steps = int(arg)
        elif opt in ("-n", "--neurons"):
            neurons = int(arg)
        elif opt in ("-l", "--lr"):
            lr = float(arg)
        elif opt in ("-g", "--gamma"):
            gamma = float(arg)
        elif opt in ("-d", "--duration"):
            duration = float(arg)
            env.config["duration"] = duration

    actor_critic_agent = A2C(env=env, training_steps=training_steps, testing_steps=testing_steps, hidden_neurons=neurons, lr=lr, gamma=gamma)
    actor_critic_baseline = A2CBaseline(env=env, training_steps=training_steps, testing_steps=testing_steps, lr=lr, gamma=gamma)
    plotter = Plotting()
    
    # Training
    print("A2C Training")
    a2c_training_start = time.time()
    training_action_distribution, training_rewards = actor_critic_agent.train_episode()
    a2c_training_end = time.time()
    a2c_training_time = a2c_training_end - a2c_training_start
    actor_critic_agent.actor_critic.save("a2c.model.h5")

    print("Baseline Training")
    baseline_start = time.time()
    actor_critic_baseline.train_model()
    baseline_end = time.time()
    baseline_time = baseline_end - baseline_start

    # Testing
    print("A2C Prediction")
    actor_critic_agent.env.config["lanes_count"] = 3
    actor_critic_agent.env.config["vehicles_count"] = 40
    actor_critic_agent.env.config["vehicles_density"] = 2

    eval_training_steps = 0
    eval_max_reward = 0
    eval_episode_rewards = []
    eval_episode_steps = []
    eval_action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    a2c_eval_start = time.time()
    for episode in range(1, num_episodes + 1):
        episode_start = time.time()
        print("Episode: ", episode)
        reward, steps, action_distribution = actor_critic_agent.model_predict()
        for key, _ in eval_action_distribution.items():
            eval_action_distribution[key] += action_distribution[key]
        eval_training_steps += steps
        eval_episode_rewards.append(reward)
        eval_episode_steps.append(steps)
        eval_max_reward = max(eval_max_reward, reward)
        print("Reward:", reward, "| Steps:", steps)
        episode_end = time.time()
        if (episode_end - episode_start > duration):
            print("Time limit reached")
            break
    a2c_eval_end = time.time()
    a2c_eval_time = a2c_eval_end - a2c_eval_start


    print("Baseline Prediction")
    actor_critic_baseline.env.config["lanes_count"] = 3
    actor_critic_baseline.env.config["vehicles_count"] = 40
    actor_critic_baseline.env.config["vehicles_density"] = 2

    baseline_max_reward = 0
    baseline_episode_rewards = []
    baseline_episode_steps = []
    baseline_total_steps = 0
    baseline_action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    baseline_eval_start = time.time()
    for episode in range(1, num_episodes + 1):
        episode_start = time.time()
        print("Episode: ", episode)
        reward, steps, action_distribution = actor_critic_baseline.model_predict()
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
    baseline_eval_end = time.time()
    baseline_eval_time = baseline_eval_end - baseline_eval_start

    """
    # Evaluate vs. DDQN:
    # Training steps
    print("DDQN training steps:", training_steps)
    print("A2C training steps:", training_steps)
    print("DDQN prediction finished in:", baseline_time, "seconds")
    print("A2C prediction finished in:", a2c_time, "seconds")

    # Evaluation with DDQN time
    print("DDQN prediction steps:", eval_training_steps)
    print("A2C prediction steps:", eval_training_steps)
    print("DDQN prediction finished in:", baseline_eval_time, "seconds")
    print("A2C prediction finished in:", a2c_eval_time, "seconds")

    # Max reward
    print("Max DDQN reward achieved:", baseline_max_reward)
    print("Max A2C reward achieved:", max_reward)
    
    # Plots
    plotter.average_episodic_plot(baseline_episode_rewards, episode_rewards, "Reward")
    plotter.episodic_plot(baseline_episode_rewards, episode_rewards, "Reward")
    plotter.episodic_plot(baseline_episode_steps, episode_steps, "Steps")
    plotter.bar_graph(baseline_action_distribution, a2c_action_distribution)

    """

    # Evaluation vs. Baseline A2C
    # Training speed
    print("Baseline prediction finished in:", baseline_time, "seconds")
    print("A2C prediction finished in:", a2c_training_time, "seconds")

    # Evaluation with baseline time
    print("Baseline prediction steps:", eval_training_steps)
    print("A2C prediction steps:", eval_training_steps)
    print("Baseline prediction finished in:", baseline_eval_time, "seconds")
    print("A2C prediction finished in:", a2c_eval_time, "seconds")

    # Max reward
    print("Max baseline reward achieved:", baseline_max_reward)
    print("Max A2C reward achieved:", eval_max_reward)

    # Plots
    plotter.average_episodic_plot(baseline_episode_rewards, eval_episode_rewards, "Reward")
    plotter.episodic_plot(baseline_episode_rewards, eval_episode_rewards, "Reward")
    plotter.episodic_plot(baseline_episode_steps, eval_episode_steps, "Steps")
    plotter.bar_graph(baseline_action_distribution, eval_action_distribution)



if __name__ == "__main__":
    main(sys.argv[1:])