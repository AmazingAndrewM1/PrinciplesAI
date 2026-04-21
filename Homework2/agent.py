# Resources that may help with the assignment
# https://github.com/MehdiShahbazi/DQN-Mountain-Car-Gymnasium
# https://firner.com/presentations/cs530/2026-Spring/

import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int = 1_024):
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng()

    def add(self, state, action_index, reward, next_state, finish):
        self.buffer.append((state, action_index, reward, next_state, finish))

    def get_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(tuple, zip(*batch, strict=True))

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Agent:
    def __init__(
        self, env: gym.Env, 
        batch_size: int = 64, 
        learning_rate: float = 0.001,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.05, 
        epsilon_decay_rate: float = 0.98,
        gamma: float = 0.95, 
        clip_grad_val: float = 5.0,
        target_network_update_frequency = 256
    ):
        self.rng = np.random.default_rng()
        self.observation_space = env.observation_space
        self.action_space = np.linspace(env.action_space.low, env.action_space.high, num=7, endpoint=True).ravel()

        self.replay_buffer = ReplayBuffer()
        self.batch_size = batch_size

        self.main_network = DQNNetwork(self.observation_space.shape[0], self.action_space.shape[0])
        self.target_network = DQNNetwork(self.observation_space.shape[0], self.action_space.shape[0])
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = epsilon_max
        self.gamma = gamma
        self.clip_grad_val = clip_grad_val
        self.target_network_update_frequency = target_network_update_frequency
        self.train_count: int = 0

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

        self.loss_history = []

        self.has_printed = False

    def choose_action_index(self, state):
        if self.rng.random() < self.epsilon:
            action_index = self.rng.integers(0, self.action_space.shape[0])
            return action_index
        
        # Coerce as tensor
        state_tensor: torch.Tensor = state if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.main_network(state_tensor)
            action_index = torch.argmax(q_values).item()
        return action_index
    
    def choose_action_from_index(self, index: int):
        return np.array([self.action_space[index]], dtype=float)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:   # Wait until enough samples are present to start training
            return

        states, action_indices, rewards, next_states, finishes = self.replay_buffer.get_batch(self.batch_size)

        states_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32)
        action_indices_tensor = torch.as_tensor(action_indices, dtype=torch.int32).unsqueeze(dim=1)  # torch.int32 required for indexing in gather() method
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
        finishes_tensor = torch.as_tensor(finishes, dtype=torch.bool)

        predicted_q = self.main_network(states_tensor)
        predicted_q_taken = predicted_q.gather(dim=1, index=action_indices_tensor).squeeze(dim=1)

        with torch.no_grad():
            q_next = self.target_network(next_states_tensor).max(dim=1).values
            target_q_next = torch.where(finishes_tensor, rewards_tensor, rewards_tensor + self.gamma * q_next)    # Thanks, Bellman!

        loss = self.loss_function(predicted_q_taken, target_q_next)
        self.loss_history.append(loss.item())

        # The learning part!
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_val)  # No exploding gradients please!
        self.optimizer.step()

        self.train_count += 1
        if self.train_count % self.target_network_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())




        
