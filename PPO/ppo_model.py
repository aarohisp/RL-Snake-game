import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class PPOModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # New layer
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # New layer
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    # def sample_action(self, state):
    #     """Sample an action based on the given state."""
    #     action_probs, _ = self.forward(state)  # Get action probabilities
    #     distribution = torch.distributions.Categorical(action_probs)  # Create categorical distribution
    #     action = distribution.sample()  # Sample an action
    #     log_prob = distribution.log_prob(action)  # Calculate log probability of the action
    #     return action.item(), log_prob  # Return action and log probability


    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

    def save(self, file_name='ppo_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class PPOTrainer:
    def __init__(self, model, lr, gamma=0.99, clip_epsilon=0.05, c1=0.5, c2=0.1): #c2=0.01 clip_epsilon = 0.2
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Weight for value loss
        self.c2 = c2  # Weight for entropy bonus

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.critic_loss_fn = nn.MSELoss()

    def compute_advantage(self, rewards, values, dones):
        advantages = []
        advantage = 0
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            td_error = reward + self.gamma * value * (1 - done) - value
            advantage = td_error + self.gamma * advantage * (1 - done)
            advantages.insert(0, advantage)
        return torch.tensor(advantages, dtype=torch.float)

    def train_step(self, states, actions, rewards, log_probs, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)  # Ensure actions are 2D
        actions = actions[:states.shape[0]]  # Ensure actions match states in size

        rewards = torch.tensor(rewards, dtype=torch.float)
        old_action_probs = torch.tensor(log_probs, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            old_action_probs = torch.unsqueeze(old_action_probs, 0)
            dones = torch.unsqueeze(dones, 0)

        # Ensure all tensors have the same batch size
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == old_action_probs.shape[0], \
            f"Shape mismatch: states {states.shape}, actions {actions.shape}, rewards {rewards.shape}, log_probs {old_action_probs.shape}"

        # Get current action probabilities and value estimates
        action_probs, state_values = self.model(states)
        state_values = state_values.squeeze()

        # Gather new action probabilities using the actions taken
        new_action_probs = action_probs.gather(1, actions).squeeze()

        # Compute the ratio
        ratio = new_action_probs / old_action_probs

        # Compute the advantage
        advantages = self.compute_advantage(rewards, state_values.detach().numpy(), dones)

        # Clip the ratio for PPO
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Critic loss (value function loss)
        value_loss = self.critic_loss_fn(rewards + self.gamma * state_values * (1 - dones), state_values)

        # Entropy for exploration bonus
        entropy_bonus = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1).mean()

        # Total loss
        loss = policy_loss + self.c1 * value_loss - self.c2 * entropy_bonus

        # Update model weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
