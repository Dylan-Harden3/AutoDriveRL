from stable_baselines3 import DQN
import numpy as np

class DQNBaseline:
    def __init__(self, env, training_steps, testing_steps):
        self.env = env
        self.training_steps = training_steps
        self.testing_steps = testing_steps
        self.model = DQN("MlpPolicy", env, verbose=1)
    
    def train_model(self):
        self.model.learn(total_timesteps=self.training_steps, progress_bar=True)
        self.model.save(f"saved models/dqn_baseline")

    def model_predict(self, model):
        total_reward = 0
        total_steps = 0
        action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        state, _ = self.env.reset()
        for step in range(1, self.testing_steps + 1):
            total_steps = step
            action, _ = model.predict(state)
            action_distribution[int(action)] += 1
            next_state, reward, done, _, _ = self.env.step(action)

            total_reward += reward

            if done:
                break

            state = next_state

        return total_reward, total_steps
