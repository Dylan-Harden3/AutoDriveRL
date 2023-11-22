from matplotlib import pyplot as plt
import numpy as np

class Plotting():

    def average_episodic_plot(self, baseline_metric_values, metric_values, metric_name):
        baseline_average_values = [sum(baseline_metric_values[:i+1]) / len(baseline_metric_values[:i+1]) for i in range(len(baseline_metric_values))]
        average_values = [sum(metric_values[:i+1]) / len(metric_values[:i+1]) for i in range(len(metric_values))]
        plt.plot(baseline_average_values, label="Baseline")
        plt.plot(average_values, label="A2C")
        plt.xlabel('Episodes')
        plt.ylabel(f'Average {metric_name} per Episode')
        plt.title(f'Average {metric_name} Over Time')
        plt.legend()
        plt.show()
    
    def episodic_plot(self, baseline_metric_values, metric_values, metric_name):
        plt.plot(baseline_metric_values, label="Baseline")
        plt.plot(metric_values, label="A2C")
        plt.xlabel('Episodes')
        plt.ylabel(f'{metric_name} per Episode')
        plt.title(f'Episode {metric_name} Over Time')
        plt.legend()
        plt.show()

    def bar_graph(self, baseline_distribution, a2c_distribution):
        actions = [0, 1, 2, 3, 4]
        baseline_values = baseline_distribution.values()
        a2c_values = a2c_distribution.values()

        x_axis = np.arange(len(actions))

        plt.bar(x_axis - 0.2, baseline_values, color ='maroon', width = 0.4, label="Baseline")
        plt.bar(x_axis + 0.2, a2c_values, color ='grey', width = 0.4, label="A2C")

        plt.xticks(x_axis, actions) 
        plt.xlabel("Actions")
        plt.ylabel("Number of Selections")
        plt.title("Action Distribution")
        plt.legend()
        plt.show()