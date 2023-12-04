import numpy as np
import random
from copy import deepcopy
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DQN():
    def __init__(self, env, training_steps, testing_steps, hidden_neurons, lr, gamma, epsilon, replay_memory_size, batch_size, update_target_every, per_alpha, num_layers):
        self.env = env
        self.training_steps = training_steps
        self.testing_steps = testing_steps
        self.hidden_neurons = hidden_neurons
        self.model = self.q_network(self.env.observation_space.shape, hidden_neurons, self.env.action_space.n, num_layers)
        self.target_model = deepcopy(self.model)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.loss_fn = keras.losses.Huber()
        self.per_alpha = per_alpha
        for layer in self.target_model.layers:
            layer.trainable = False

    def q_network(self, input_shape, hidden_neurons, num_actions, num_layers):
        inputs = keras.Input(input_shape, name="input_layer")
        flattened = layers.Flatten()(inputs)
        for _ in range(num_layers):
            if _ == 0:
                hidden = layers.Dense(hidden_neurons, activation="relu")(flattened)
            else:
                hidden = layers.Dense(hidden_neurons, activation="relu")(hidden)
        q_values = layers.Dense(num_actions, activation="linear")(hidden)
        model = keras.Model(inputs=inputs, outputs=q_values)
        return model

    def epsilon_greedy(self, state, step):
        state = np.expand_dims(state, axis=0)
        q_values = self.model(state)
        a = tf.argmax(q_values, axis=1)[0]
        num_actions = self.env.action_space.n
        epsilon = np.clip(self.initial_epsilon - (self.initial_epsilon) * (current_step / 5000), 0.0, self.initial_epsilon) # decay epsilon over first 5000 steps
        return np.array([1 - epsilon + (epsilon / num_actions) if i == a else epsilon / num_actions for i in range(num_actions)])

    def compute_target_values(self, next_states, rewards, dones):
        return rewards + self.gamma * (1-dones) * tf.reduce_max(self.target_model(next_states), axis=1)

    def replay(self):
        if len(self.replay_memory) >= self.batch_size: 
            states, actions, rewards, next_states, dones = zip(*list(self.replay_memory))

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            current_q = self.model(states)
            current_q = tf.gather_nd(current_q, tf.stack([tf.range(tf.shape(current_q)[0]), actions], axis=1))

            target_q = self.compute_target_values(next_states, rewards, dones)

            td_errors = tf.abs(target_q-current_q)
            probs = tf.pow(td_errors, self.per_alpha) / tf.reduce_sum(tf.pow(td_errors, self.per_alpha))
            probs = tf.reshape(probs, [1, -1])
            indices = tf.squeeze(tf.random.categorical(probs, self.batch_size))
                
            # sample minibatch
            states = tf.gather(states, indices)
            actions = tf.gather(actions, indices)
            rewards = tf.gather(rewards, indices)
            next_states = tf.gather(next_states, indices)
            dones = tf.gather(dones, indices)
            
            current_q = self.model(states)
            current_q = tf.gather_nd(current_q, tf.stack([tf.range(tf.shape(current_q)[0]), actions], axis=1))

            with tf.GradientTape() as tape:
                target_q = self.compute_target_values(next_states, rewards, dones)
                loss_q = self.loss_fn(current_q, target_q)

            grads = tape.gradient(loss_q, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_episode(self):
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
            action = np.random.choice(np.arange(self.env.action_space.n), p=self.epsilon_greedy(state, step_number))
            next_state, reward, done, _, _ = self.env.step(action)
            rewards.append(reward)
            action_distribution[int(action)] += 1
            self.replay_memory.append((state, action, reward, next_state, done))
            self.replay()
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
            if step_number % self.update_target_every == 0:
                self.target_model.set_weights(self.model.get_weights())
    
        self.model.save(f"saved models/dqn_model_{self.per_alpha}.h5")
        return action_distribution, episode_rewards

    def model_predict(self, model):
        total_reward = 0
        total_steps = 0
        state, _ = self.env.reset()
        state = np.expand_dims(state, axis=0)
        for step in range(1, self.testing_steps + 1):
            total_steps = step
            action = np.argmax(model.predict(state))
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = np.expand_dims(next_state, axis=0)

            total_reward += reward

            if done:
                break

            state = next_state

        return total_reward, total_steps