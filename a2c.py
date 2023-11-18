import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
from torch.optim import Adam
from lib import plotting

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class A2C():
    def __init__(self, env, layers, lr, gamma):
        # Create actor-critic network
        self.actor_critic = self.a2c_network(env.observation_space.shape[0], env.action_space.n, layers)
        self.policy = self.create_greedy_policy()
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.discount_factor = gamma

    def a2c_network(self, num_inputs, num_hidden, num_actions):
        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        model = keras.Model(inputs=inputs, outputs=[action, critic])
        return model

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            return tf.math.argmax(self.actor_critic(state)[0]).detach().numpy()

        return policy_fn

    def select_action(self, state):
        """
        Selects an action given state.

        Returns:
            The selected action (as an int)
            The probability of the selected action (as a tensor)
            The critic's value estimate (as a tensor)
        """
        state = tf.convert_to_tensor(state, dtype=torch.float32)
        probs, value = self.actor_critic(state)

        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)

        return action, probs[action], value

    def update_actor_critic(self, advantage, prob, value, tape):
        """
        Performs actor critic update.

        args:
            advantage: Advantage of the chosen action (tensor).
            prob: Probability associated with the chosen action (tensor).
            value: Critic's state value estimate (tensor).
        """
        # Compute loss
        actor_loss = self.actor_loss(advantage.detach(), prob).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()

        loss = actor_loss + critic_loss

        grads = tape.gradient(loss, self.actor_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

    def train_episode(self):
        """
        Run a single episode of the A2C algorithm.

        Use:
            self.select_action(state): Sample an action from the policy.
            self.step(action): Perform an action in the env.
            self.options.gamma: Gamma discount factor.
            self.actor_critic(state): Returns the action probabilities and
                the critic's estimate at a given state.
            torch.as_tensor(state, dtype=torch.float32): Converts a numpy array
                'state' to a tensor.
            self.update_actor_critic(advantage, prob, value): Update actor critic. 
        """

        state, _ = self.env.reset()
        with tf.GradientTape() as tape:
            for _ in range(self.options.steps):
                ################################
                #   YOUR IMPLEMENTATION HERE   #
                # Run update_actor_critic()    #
                # only ONCE at EACH step in    #
                # an episode.                  # 
                ################################
                action, prob, value = self.select_action(state)
                next_state, reward, done, _ = self.step(action)

                next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)
                _, next_value = self.actor_critic(next_state_tensor)

                if done:
                    next_value = next_value * 0
                
                advantage = reward + (self.discount_factor * next_value) - value

                self.update_actor_critic(advantage, prob, value, tape)

                if done:
                    break

                state = next_state


    def actor_loss(self, advantage, prob):
        """
        The policy gradient loss function.
        Note that you are required to define the Loss^PG
        which should be the integral of the policy gradient.

        args:
            advantage: Advantage of the chosen action.
            prob: Probability associated with the chosen action.

        Use:
            torch.log: Element-wise logarithm.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        loss = -tf.math.log(prob) * advantage
        return loss

    def critic_loss(self, advantage, value):
        """
        The integral of the critic gradient

        args:
            advantage: Advantage of the chosen action.
            value: Critic's state value estimate.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        loss = -advantage * value
        return loss

    def __str__(self):
        return "A2C"
