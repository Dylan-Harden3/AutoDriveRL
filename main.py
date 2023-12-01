import numpy as np
import gymnasium as gym
import time
import getopt, sys
import os
import csv

from agents.ddqn import DDQN
from agents.a2c import A2C
from baselines.a2c_baseline import A2CBaseline
from helpers.evaluation import *
from helpers.plotting import Plotting
from tensorflow.keras import models
from stable_baselines3 import A2C as A2C3



def main(argv):
    # Hyperparameters
    training_steps = 10000
    testing_episodes = 10
    testing_steps = 1000
    neurons = 256
    lr = 0.001
    gamma = 0.8
    duration = 90
    epsilon = 0.9
    replay_memory_size = 15000
    update_target_every = 50
    batch_size = 32

    # Environment configuration parameters
    env = gym.make('highway-v0', render_mode='human')
    #env.config["lanes_count"] = 5
    env.config["duration"] = duration
    #env.config["lane_change_reward"] = 0
    env.config["right_lane_reward"] = 0.05
    env.config["collision_reward"] = -5
    env.config["high_speed_reward"] = 0.8
    env.config["reward_speed_range"] = [30, 40]
    #env.config["vehicles_density"] = 2
    env.config["observation"] = {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "heading"]
    }
    env.reset()
    
    try:
        opts, _ = getopt.getopt(argv, "he:s:t:n:l:g:d:E:m:N:B", ["help=", "episodes=", "steps=", "testing steps=" "neurons=", 
                                                                "learning rate=", "gamma=", "duration=", "epsilon=", 
                                                                "replay memory size=", "update target every=", "batch size="])
    except getopt.GetoptError:
        print('Usage: main.py [-h <help>] [-e <episodes>] [-s <steps>] [-t <testing steps>] [-n <neurons>] [-l <learning rate>] [-g <gamma>] [-d <duration>] [-E <epsilon>] [-m <replay memory size>] [-N <target update interval>] [-B <batch size>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: main.py [-e <episodes>] [-s <steps>] [-t <testing steps>] [-n <neurons>] [-l <lr>] [-g <gamma>] [-d <duration>] [-E <epsilon>] [-m <replay memory size>] [-N <target update interval>] [-B <batch size>]')
            sys.exit()
        elif opt in ("-e", "--episodes"):
            testing_episodes = int(arg)
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
        elif opt in ("-E", "--epsilon"):
            epsilon = float(arg)
        elif opt in ("-m", "--replay_memory_size"):
            replay_memory_size = int(arg)
        elif opt in ("-N", "--update_target_every"):
            update_target_every = int(arg)
        elif opt in ("-B", "--batch_size"):
            batch_size = int(arg)

    actor_critic = A2C(env=env, training_steps=training_steps, testing_steps=testing_steps, hidden_neurons=neurons, lr=lr, gamma=gamma)
    actor_critic_baseline = A2CBaseline(env=env, training_steps=training_steps, testing_steps=testing_steps, lr=lr, gamma=gamma)
    dqn = DDQN(env, training_steps, testing_steps, neurons, lr, gamma, epsilon, replay_memory_size, batch_size, update_target_every)
    plotter = Plotting()

    # Training A2C
    print("A2C Training")
    a2c_training_action_distribution, a2c_training_rewards, a2c_training_max_reward = actor_critic.train_episode()
    env.reset()

    # # Training DQN
    print("DDQN Training")
    ddqn_training_action_distribution, ddqn_training_rewards, ddqn_training_max_reward = dqn.train_episode()
    env.reset()

    # Training A2C baseline
    print("Baseline Training")
    actor_critic_baseline.train_model()
    env.reset()

    # # Testing A2C
    # a2c_model = models.load_model("saved models/a2c_model.h5")
    # print("A2C Prediction")

    # a2c_eval_start = time.time()
    # a2c_eval_max_reward, a2c_eval_episode_rewards, a2c_eval_episode_steps, a2c_eval_action_distribution = prediction(episodes=testing_episodes, agent=actor_critic, duration=duration, model=a2c_model)
    # a2c_eval_end = time.time()
    # a2c_eval_time = a2c_eval_end - a2c_eval_start

    # # Testing DDQN
    # ddqn_model = models.load_model("saved models/ddqn_model.h5")
    # print("DDQN Prediction")

    # ddqn_eval_start = time.time()
    # ddqn_eval_max_reward, ddqn_eval_episode_rewards, ddqn_eval_episode_steps, ddqn_eval_action_distribution = prediction(episodes=testing_episodes, agent=dqn, duration=duration, model=ddqn_model)
    # ddqn_eval_end = time.time()
    # ddqn_eval_time = ddqn_eval_end - ddqn_eval_start

    # # Testing baseline A2C
    # a2c_baseline_model = A2C3.load("saved models/a2c_baseline")
    # print("A2C Baseline Prediction")

    # a2c_baseline_eval_start = time.time()
    # a2c_baseline_prediction_steps, a2c_baseline_max_reward, a2c_baseline_episode_rewards, a2c_baseline_episode_steps, a2c_baseline_action_distribution = prediction(episodes=testing_episodes, agent=actor_critic_baseline, duration=duration, model=a2c_baseline_model)
    # a2c_baseline_eval_end = time.time()
    # a2c_baseline_eval_time = a2c_baseline_eval_end - a2c_baseline_eval_start
    
    # Evaluation
    # Training max reward
    print("Max A2C training reward achieved:", a2c_training_max_reward)
    print("Max DDQN training reward achieved:", ddqn_training_max_reward)

    # Prediction time
    # print("A2C prediction finished in:", a2c_eval_time, "seconds")
    # print("DDQN prediction finished in:", ddqn_eval_time, "seconds")
    # print("A2C Baseline prediction finished in:", a2c_baseline_eval_time, "seconds")

    # # Evaluation Max reward
    # print("Max A2C reward achieved:", a2c_eval_max_reward)
    # print("Max DDQN reward achieved:", ddqn_eval_max_reward)
    # print("Max A2C baseline reward achieved:", a2c_baseline_max_reward)
    
    # Plots
    # Plotting DDQN vs. A2C Training
    plotter.average_episodic_plot(a2c_training_rewards, ddqn_training_rewards, "Reward", "A2C", "DQN")
    plotter.episodic_plot(a2c_training_rewards, ddqn_training_rewards, "Reward", "A2C", "DQN")
    plotter.bar_graph(a2c_training_action_distribution, ddqn_training_action_distribution, "A2C", "DQN")
    plotter.write_rewards_to_csv(a2c_training_rewards, "A2C")
    plotter.write_rewards_to_csv(ddqn_training_rewards, "DDQN")
    plotter.write_actions_to_csv(a2c_training_action_distribution, "A2C")
    plotter.write_actions_to_csv(ddqn_training_action_distribution, "DDQN")

    # Plotting A2C vs. Baseline Testing
    # plotter.episodic_plot(a2c_baseline_episode_rewards, a2c_eval_episode_rewards, "Reward", "Baseline", "A2C")
    # plotter.episodic_plot(a2c_baseline_episode_steps, a2c_eval_episode_steps, "Steps", "Baseline", "A2C")
    # plotter.bar_graph(a2c_baseline_action_distribution, a2c_eval_action_distribution, "Baseline", "A2C")




if __name__ == "__main__":
    main(sys.argv[1:])