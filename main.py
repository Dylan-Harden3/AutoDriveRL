import numpy as np
import gymnasium as gym
import time
import getopt, sys
import os
import matplotlib.pyplot as plt


# from agents.ddqn import DDQN
from agents.a2c import A2C
from baselines.a2c_baseline import A2CBaseline
from helpers.evaluation import *
from helpers.plotting import Plotting
from tensorflow.keras import models
from stable_baselines3 import A2C as A2C3



def main(argv):
    # Hyperparameters
    training_steps = 50000
    num_episodes = 10
    testing_steps = 1000
    neurons = 256
    lr = 0.0001
    gamma = 0.95
    duration = 150

    # Environment configuration parameters
    env = gym.make('highway-v0')
    env.config["lanes_count"] = 5
    env.config["duration"] = duration
    env.config["lane_change_reward"] = 0
    env.config["right_lane_reward"] = 0.1
    env.config["collision_reward"] = -10
    env.config["high_speed_reward"] = 0.8
    env.config["reward_speed_range"] = [10, 40]
    env.config["vehicles_count"] = 50
    env.config["vehicles_density"] = 2
    config = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (128, 64),
           "stack_size": 4,
           "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
       },
       "policy_frequency": 2
    }
    env.configure(config)
    obs, info = env.reset()
    try:
        opts, _ = getopt.getopt(argv, "he:s:t:n:l:g:d:", ["help=", "episodes=", "steps=", "testing steps=" "neurons=", 
                                                        "learning rate=", "gamma=", "duration="])
    except getopt.GetoptError:
        print('Usage: main.py [-h <help>] [-e <episodes>] [-s <steps>] [-t <testing steps>] [-n <neurons>] [-l <learning rate>] [-g <gamma>] [-d <duration>]')
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
    
    # Training A2C
    print("A2C Training")
    a2c_training_start = time.time()
    training_action_distribution, training_rewards = actor_critic_agent.train_episode()
    a2c_training_end = time.time()
    a2c_training_time = a2c_training_end - a2c_training_start

    # Training baseline
    print("Baseline Training")
    baseline_start = time.time()
    actor_critic_baseline.train_model()
    baseline_end = time.time()
    baseline_time = baseline_end - baseline_start

    # Testing A2C
    model = models.load_model("saved models/a2c_network.h5")
    print("A2C Prediction")
    actor_critic_agent.env.config["lanes_count"] = 5
    actor_critic_agent.env.config["vehicles_count"] = 60
    actor_critic_baseline.env.config["vehicles_density"] = 2

    a2c_eval_start = time.time()
    eval_prediction_steps, eval_max_reward, eval_episode_rewards, eval_episode_steps, eval_action_distribution = prediction(episodes=num_episodes, agent=actor_critic_agent, duration=duration, model=model)
    a2c_eval_end = time.time()
    a2c_eval_time = a2c_eval_end - a2c_eval_start

    # Testing baseline
    model = A2C3.load("saved models/a2c_baseline")
    print("Baseline Prediction")
    actor_critic_baseline.env.config["lanes_count"] = 5
    actor_critic_baseline.env.config["vehicles_count"] = 60
    actor_critic_baseline.env.config["vehicles_density"] = 2

    baseline_eval_start = time.time()
    baseline_prediction_steps, baseline_max_reward, baseline_episode_rewards, baseline_episode_steps, baseline_action_distribution = prediction(episodes=num_episodes, agent=actor_critic_baseline, duration=duration, model=model)
    baseline_eval_end = time.time()
    baseline_eval_time = baseline_eval_end - baseline_eval_start

    # Evaluation vs. Baseline A2C
    # Evaluation with baseline time
    print("Baseline prediction steps:", baseline_prediction_steps)
    print("A2C prediction steps:", eval_prediction_steps)
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