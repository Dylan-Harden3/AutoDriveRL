import numpy as np
import gymnasium as gym
import time
from collections import deque
import random
from copy import deepcopy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DDQN():
    def __init__(self, env, training_steps, testing_steps, hidden_neurons, lr, gamma, epsilon, replay_memory_size, batch_size, update_target_every):
        self.env = env
        self.training_steps = training_steps
        self.testing_steps = testing_steps
        self.hidden_neurons = hidden_neurons
        self.model = self.q_network(self.env.observation_space.shape, hidden_neurons, self.env.action_space.n)
        self.target_model = deepcopy(self.model)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.loss_fn = keras.losses.Huber()

        for layer in self.target_model.layers:
            layer.trainable = False

    def q_network(self, input_shape, hidden_neurons, num_actions):
        inputs = keras.Input(input_shape, name="input_layer")
        flattened = layers.Flatten()(inputs)
        hidden = layers.Dense(hidden_neurons, activation="relu")(flattened)
        hidden = layers.Dense(hidden_neurons, activation="relu")(hidden)
        q_values = layers.Dense(num_actions, activation="linear")(hidden)
        model = keras.Model(inputs=inputs, outputs=q_values)
        return model

        
    def decay_epsilon(self, current_step, total_steps, initial_epsilon):
        """
        Decay function for epsilon.

        Args:
        - current_step (int): The current step or iteration.
        - total_steps (int): The total number of steps over which decay will occur.
        - initial_epsilon (float): The initial epsilon value.
        - final_epsilon (float): The final epsilon value.

        Returns:
        - float: The decayed epsilon value.
        """
        epsilon = initial_epsilon - (initial_epsilon) * (current_step / total_steps)
        return np.clip(epsilon, 0.0, initial_epsilon)

    def epsilon_greedy(self, state, step):
        state = np.expand_dims(state, axis=0)
        q_values = self.model(state)
        a = tf.argmax(q_values, axis=1)[0]
        num_actions = self.env.action_space.n
        epsilon = self.decay_epsilon(step, self.training_steps, self.initial_epsilon)
        return np.array([1 - epsilon + (epsilon / num_actions) if i == a else epsilon / num_actions for i in range(num_actions)])


    def compute_target_values(self, next_states, rewards, dones):
        return rewards + self.gamma * (1-dones) * tf.reduce_max(self.target_model(next_states), axis=1)

    def replay(self):
        if len(self.replay_memory) >= self.batch_size:
            minibatch = random.sample(self.replay_memory, self.batch_size)
           
            states, actions, rewards, next_states, dones = zip(*minibatch)

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            current_q = self.model(states)
            current_q = tf.gather_nd(current_q, tf.stack([tf.range(tf.shape(current_q)[0]), actions], axis=1))

            with tf.GradientTape() as tape:
                target_q = self.compute_target_values(next_states, rewards, dones)
                loss_q = self.loss_fn(current_q, target_q)

            grads = tape.gradient(loss_q, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation. Finds the optimal greedy policy
        while following an epsilon-greedy policy.

        Use:
            self.epsilon_greedy(state): return probabilities of actions.
            np.random.choice(array, p=prob): sample an element from 'array' based on their corresponding
                probabilites 'prob'.
            self.memorize(state, action, reward, next_state, done): store the transition in the replay buffer
            self.update_target_model(): copy weights from model to target_model
            self.replay(): TD learning for q values on past transitions
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps
        """

        # Reset the environment
        state, _ = self.env.reset()
        rewards = []
        action_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        episode_rewards = []
        episode = 1
        done = False
        rewards = []
        for step_number in range(1, self.training_steps+1):
            if done:
                episode_rewards.append(sum(rewards))
                rewards = []
                print(f"Episode: {episode} Reward {episode_rewards[-1]}")
                episode += 1
            # print("Step:", step_number, "out of", self.training_steps)
            action = np.random.choice(np.arange(self.env.action_space.n), p=self.epsilon_greedy(state, step_number))
            next_state, reward, done, _, _ = self.env.step(action)
            rewards.append(reward)
            action_distribution[int(action)] += 1
            # print("Reward:", reward)
            self.memorize(state, action, reward, next_state, done)
            self.replay()
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
            if step_number % self.update_target_every == 0:
                self.update_target_model()
        return action_distribution, episode_rewards