import numpy as np
import gymnasium as gym
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class A2C():
    def __init__(self, env, steps, hidden_neurons, lr, gamma):
        self.env = env
        self.steps = steps
        self.actor_critic = self.a2c_network(self.env.observation_space.shape[0], hidden_neurons, self.env.action_space.n)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.discount_factor = gamma


    def a2c_network(self, num_inputs, num_hidden, num_actions):
        inputs = layers.Input(num_inputs, name="input_layer")
        common = layers.Dense(num_hidden, activation="relu", name="common_layer")(inputs)
        actor = layers.Dense(num_actions, activation="softmax", name="actor_layer")(common)
        critic = layers.Dense(1, name="critic_layer")(common)

        model = keras.Model(inputs=inputs, outputs=[actor, critic])
        return model


    def select_action(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float64)
        ego_state = state[0]
        ego_state = tf.expand_dims(ego_state, 0)
        ego_probs, value = self.actor_critic(ego_state)

        action = np.random.choice(self.env.action_space.n, p=np.squeeze(ego_probs))

        return action, ego_probs[0, action], value[0, 0]


    def update_actor_critic(self, advantage, prob, value, tape):
        actor_loss = self.actor_loss(advantage, prob)
        critic_loss = self.critic_loss(advantage, value)
        loss = actor_loss + critic_loss

        grads = tape.gradient(loss, self.actor_critic.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))


    def train_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        total_steps = 0
        for step in range(1, self.steps + 1):
            with tf.GradientTape() as tape:
                total_steps = step
                action, prob, value = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                next_state_tensor = tf.convert_to_tensor(next_state)
                _, next_value = self.actor_critic(next_state_tensor)
                next_value = next_value[0, 0]

                if done:
                    next_value = next_value * 0
                
                advantage = (reward + (self.discount_factor * next_value)) - value

                self.update_actor_critic(advantage, prob, value, tape)
                total_reward += reward

                if done:
                    break

                state = next_state

        return total_reward, total_steps


    def actor_loss(self, advantage, prob):
        loss = -tf.math.log(prob) * advantage
        return loss


    def critic_loss(self, advantage, value):
        loss = -advantage * value
        return loss


    def __str__(self):
        return "A2C"
