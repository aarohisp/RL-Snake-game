import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Additional hidden layers for better representation
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        # Actor and Critic heads
        self.actor = nn.Linear(hidden_size, output_size)  # Policy output
        self.critic = nn.Linear(hidden_size, 1)  # Value output
        
        # Optional: Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first layer
        x = torch.relu(self.fc2(x))
       
        
        policy = torch.softmax(self.actor(x), dim=-1)  # Action probabilities
        value = self.critic(x)  # State value
        return policy, value


class A3CTrainer:
    def __init__(self, model, lr, gamma, optimizer_name='adam'):
        self.model = model
        self.gamma = gamma
        self.optimizer = getattr(optim, optimizer_name.capitalize())(model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Loss function for the critic

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(0)  # Ensure actions have batch dimension
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # Get predictions
        policies, values = self.model(states)
        _, next_values = self.model(next_states)

        # Debug: Print shapes
        # print("states shape:", states.shape)
        # print("next_states shape:", next_states.shape)
        # print("policies shape:", policies.shape)
        # print("values shape:", values.shape)
        # print("next_values shape:", next_values.shape)

        # Compute advantage and target value
        advantages = rewards + self.gamma * next_values.squeeze() * (1 - dones) - values.squeeze()
        critic_loss = advantages.pow(2).mean()

        # Compute actor loss using negative log-likelihood
        log_probs = torch.log(policies.unsqueeze(0).gather(1, actions))
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Total loss
        loss = actor_loss + critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
