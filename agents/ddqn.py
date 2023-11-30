# import numpy as np
# import gymnasium as gym
# import time

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


# # env = gym.make('highway-v0', render_mode='human')

# # class QFunction(nn.Module):
# #     """
# #     Q-network definition.
# #     """

# #     def __init__(
# #         self,
# #         obs_dim,
# #         act_dim,
# #         hidden_sizes,
# #     ):
# #         super().__init__()
# #         sizes = [obs_dim] + hidden_sizes + [act_dim]
# #         self.layers = nn.ModuleList()
# #         for i in range(len(sizes) - 1):
# #             self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

# #     def forward(self, obs):
# #         x = torch.cat([obs], dim=-1)
# #         for i in range(len(self.layers) - 1):
# #             x = F.relu(self.layers[i](x))
# #         return self.layers[-1](x).squeeze(dim=-1)

# # class DDQN():
# #     def train_episode(self):
# #         state, _ = env.reset()
# #         done = truncated = False
# #         for _ in range(self.options.steps):
# #             action = ... # Your agent code here
# #             next_state, reward, done, _, _ = env.step(action)

# #             if done:
# #                 break

# #             state = next_state


# class QFunction(nn.Module):
#     """
#     Q-network definition.
#     """

#     def __init__(
#         self,
#         obs_dim,
#         act_dim,
#         hidden_sizes,
#     ):
#         super().__init__()
#         sizes = [obs_dim] + hidden_sizes + [act_dim]
#         self.layers = nn.ModuleList()
#         for i in range(len(sizes) - 1):
#             self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

#     def forward(self, obs):
#         x = torch.cat([obs], dim=-1)
#         for i in range(len(self.layers) - 1):
#             x = F.relu(self.layers[i](x))
#         return self.layers[-1](x).squeeze(dim=-1)


# class DQN(AbstractSolver):
#     def __init__(self, env, eval_env, options):
#         assert str(env.action_space).startswith("Discrete") or str(
#             env.action_space
#         ).startswith("Tuple(Discrete"), (
#             str(self) + " cannot handle non-discrete action spaces"
#         )
#         super().__init__(env, eval_env, options)
#         # Create Q-network
#         self.model = QFunction(
#             env.observation_space.shape[0],
#             env.action_space.n,
#             self.options.layers,
#         )
#         # Create target Q-network
#         self.target_model = deepcopy(self.model)
#         # Set up the optimizer
#         self.optimizer = AdamW(
#             self.model.parameters(), lr=self.options.alpha, amsgrad=True
#         )
#         # Define the loss function
#         self.loss_fn = nn.SmoothL1Loss()

#         # Freeze target network parameters
#         for p in self.target_model.parameters():
#             p.requires_grad = False

#         # Replay buffer
#         self.replay_memory = deque(maxlen=options.replay_memory_size)

#         # Number of training steps so far
#         self.n_steps = 0

#     def update_target_model(self):
#         # Copy weights from model to target_model
#         self.target_model.load_state_dict(self.model.state_dict())

#     def epsilon_greedy(self, state):
#         """
#         Apply an epsilon-greedy policy based on the given Q-function approximator and epsilon.

#         Returns:
#             The probabilities (as a Numpy array) associated with each action for 'state'.

#         Use:
#             self.env.action_space.n: Number of avilable actions
#             self.torch.as_tensor(state): Convert Numpy array ('state') to a tensor
#             self.model(state): Returns the predicted Q values at a 
#                 'state' as a tensor. One value per action.
#             torch.argmax(values): Returns the index corresponding to the highest value in
#                 'values' (a tensor)
#         """
#         # Don't forget to convert the states to torch tensors to pass them through the network.
#         nA = self.env.action_space.n
#         predictions = self.model(torch.as_tensor(state))
#         a = torch.argmax(predictions)
#         return np.array([1 - self.options.epsilon + (self.options.epsilon/nA) if i == a else self.options.epsilon / nA for i in range(nA)])


#     def compute_target_values(self, next_states, rewards, dones, istest=False):
#         """
#         Computes the target q values.

#         Returns:
#             The target q value (as a tensor) of shape [len(next_states)]
#         """
#         y_j = []
#         r = min(self.options.batch_size, next_states.shape[0])
        
#         for i in range(r):
#             if i >= next_states.shape[0] or i >= dones.shape[0] or i >= rewards.shape[0]:
#                 break
#             if int(dones[i]) == 1:
#                 y_j.append(rewards[i])
#             else:
#                 y_j.append(rewards[i] + self.options.gamma * torch.max(self.target_model(next_states[i])))

#         return torch.tensor(y_j)


#     def replay(self):
#         """
#         TD learning for q values on past transitions.

#         Use:
#             self.target_model(state): predicted q values as an array with entry
#                 per action
#         """
#         if len(self.replay_memory) > self.options.batch_size:
#             minibatch = random.sample(self.replay_memory, self.options.batch_size)
#             minibatch = [
#                 np.array(
#                     [
#                         transition[idx]
#                         for transition, idx in zip(minibatch, [i] * len(minibatch))
#                     ]
#                 )
#                 for i in range(5)
#             ]
#             states, actions, rewards, next_states, dones = minibatch
#             # Convert numpy arrays to torch tensors
#             states = torch.as_tensor(states, dtype=torch.float32)
#             actions = torch.as_tensor(actions, dtype=torch.float32)
#             rewards = torch.as_tensor(rewards, dtype=torch.float32)
#             next_states = torch.as_tensor(next_states, dtype=torch.float32)
#             dones = torch.as_tensor(dones, dtype=torch.float32)

#             # Current Q-values
#             current_q = self.model(states)
#             # Q-values for actions in the replay memory
#             current_q = torch.gather(
#                 current_q, dim=1, index=actions.unsqueeze(1).long()
#             ).squeeze(-1)

#             with torch.no_grad():
#                 target_q = self.compute_target_values(next_states, rewards, dones)

#             # Calculate loss
#             loss_q = self.loss_fn(current_q, target_q)

#             # Optimize the Q-network
#             self.optimizer.zero_grad()
#             loss_q.backward()
#             torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
#             self.optimizer.step()

#     def memorize(self, state, action, reward, next_state, done):
#         self.replay_memory.append((state, action, reward, next_state, done))

#     def train_episode(self):
#         """
#         Perform a single episode of the Q-Learning algorithm for off-policy TD
#         control using a DNN Function Approximation. Finds the optimal greedy policy
#         while following an epsilon-greedy policy.

#         Use:
#             self.epsilon_greedy(state): return probabilities of actions.
#             np.random.choice(array, p=prob): sample an element from 'array' based on their corresponding
#                 probabilites 'prob'.
#             self.memorize(state, action, reward, next_state, done): store the transition in the replay buffer
#             self.update_target_model(): copy weights from model to target_model
#             self.replay(): TD learning for q values on past transitions
#             self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
#                 target estimator every N steps
#         """

#         # Reset the environment
#         state, _ = self.env.reset()

#         for _ in range(self.options.steps):
#             action = np.random.choice(np.arange(self.env.action_space.n), p=self.epsilon_greedy(state))
#             next_state, reward, done, __ = self.step(action)
#             self.memorize(state, action, reward, next_state, done)
#             self.replay()
#             if done:
#                 break
#             else:
#                 state = next_state
#             self.n_steps += 1
#             if self.n_steps % self.options.update_target_estimator_every == 0:
#                 self.update_target_model()

#     def __str__(self):
#         return "DQN"

#     def plot(self, stats, smoothing_window, final=False):
#         plotting.plot_episode_stats(stats, smoothing_window, final=final)

#     def create_greedy_policy(self):
#         """
#         Creates a greedy policy based on Q values.


#         Returns:
#             A function that takes an observation as input and returns a greedy
#             action
#         """

#         def policy_fn(state):
#             state = torch.as_tensor(state, dtype=torch.float32)
#             q_values = self.model(state)
#             return torch.argmax(q_values).detach().numpy()

#         return policy_fn
