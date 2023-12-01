from matplotlib import pyplot as plt
import numpy as np
import csv


class Plotting():

    def average_episodic_plot(self, metric_values1, metric_values2, metric_name, label1, label2):
        average_values1 = [sum(metric_values1[:i+1]) / len(metric_values1[:i+1]) for i in range(len(metric_values1))]
        average_values2 = [sum(metric_values2[:i+1]) / len(metric_values2[:i+1]) for i in range(len(metric_values2))]
        plt.plot(average_values1, label=label1)
        plt.plot(average_values2, label=label2)
        plt.xlabel('Episodes')
        plt.ylabel(f'Average {metric_name} per Episode')
        plt.title(f'Average {metric_name} Over Time')
        plt.legend()
        plt.savefig(f'{label1}_vs_{label2}_{metric_name.lower()}_average_plot.png')
        plt.show()
    
    def episodic_plot(self, metric_values1, metric_values2, metric_name, label1, label2):
        plt.plot(metric_values1, label=label1)
        plt.plot(metric_values2, label=label2)
        plt.xlabel('Episodes')
        plt.ylabel(f'{metric_name} per Episode')
        plt.title(f'Episode {metric_name} Over Time')
        plt.legend()
        plt.savefig(f'{label1}_vs_{label2}_{metric_name.lower()}_episodic_plot.png')
        plt.show()

    def bar_graph(self, metric_values1, metric_values2, label1, label2):
        actions = ["0: Lane Left", "1: Idle", "2: Lane Right", "3: Faster", "4: Slower"]
        values1 = metric_values1.values()
        values2 = metric_values2.values()

        x_axis = np.arange(len(actions))

        plt.bar(x_axis - 0.2, values1, color ='maroon', width = 0.4, label=label1)
        plt.bar(x_axis + 0.2, values2, color ='grey', width = 0.4, label=label2)

        plt.xticks(x_axis, actions) 
        plt.xlabel("Actions")
        plt.ylabel("Number of Selections")
        plt.title("Action Distribution")
        plt.legend()
        plt.savefig(f'{label1}_vs_{label2}_action_distribution.png')
        plt.show()

    def write_rewards_to_csv(self, rewards, label):
        episodes = np.arange(1, len(rewards) + 1)
        with open(f'{label}_episode_rewards.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Reward'])
            for episode, reward in zip(episodes, rewards):
                writer.writerow([episode, reward])

    def write_actions_to_csv(self, action_distribution, label):
        with open(f'{label}_action_distribution.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Action', 'Selections'])
            for episode, reward in action_distribution.items():
                writer.writerow([episode, reward])