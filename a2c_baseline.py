import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env



class A2CBaseline():
    def __init__(self, env, steps, lr, gamma):
        self.env = env
        self.steps = steps
        self.model = A2C("MlpPolicy", env, learning_rate=lr, gamma=gamma, use_rms_prop=False)

    def model_learn(self):
        self.model.learn(total_timesteps=self.steps, progress_bar=True)
    
    def train_episode(self):
        total_reward = 0
        total_steps = 0
        action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        state, _ = self.env.reset()
        for step in range(1, self.steps + 1):
            total_steps = step
            action, _ = self.model.predict(state)
            action_distribution[int(action)] += 1
            next_state, reward, done, _, _ = self.env.step(action)

            total_reward += reward

            if done:
                break

            state = next_state

        return total_reward, total_steps, action_distribution