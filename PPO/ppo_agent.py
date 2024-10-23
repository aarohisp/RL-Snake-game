import torch
import random
import numpy as np
import os
from collections import deque
from game import SnakeGameAI, Direction, Point
from ppo_model import PPOModel, PPOTrainer  # Import your PPO model and trainer here
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
MAX_MEMORY = 100_000
BATCH_SIZE = 64 #10000
LR = 0.001 #0.001 
EPISODE_STEPS = 8192  # PPO typically collects many steps before updating

class Agent:
    def __init__(self, optimizer_name='adam'):
        self.n_games = 0
        #self.epsilon = 0  # randomness (could be removed for PPO)
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = PPOModel(11, 256, 3)  
        self.trainer = PPOTrainer(self.model, lr=LR, gamma=self.gamma)
        self.episode_steps = EPISODE_STEPS
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def store_experience(self, state, action, reward, log_prob, done):
        """Store the experience for PPO updates"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear_memory(self):
        """Clear memory after training"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

    def get_action(self, state):
        final_move = [0, 0, 0]
        log_prob = None

        # Exploration vs exploitation
        if random.randint(0, 200):
            move = random.randint(0, 2)
            final_move[move] = 1
            log_prob = torch.log(torch.tensor(1/3))  # Log probability for random action
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)

            with torch.no_grad():
                # Assuming the model returns action_probs as the first element of the tuple
                action_probs, _ = self.model(state0)  # Extract only action probabilities

            # Sample an action from the action distribution
            action_distribution = torch.distributions.Categorical(action_probs)
            move = action_distribution.sample()  # Sample an action
            final_move[move.item()] = 1
            log_prob = action_distribution.log_prob(move)  # Get log probability of the action

        return final_move, log_prob



    def train(self):
        if len(self.states) >= self.episode_steps:
            self.trainer.train_step(
                states=self.states, 
                actions=self.actions, 
                rewards=self.rewards, 
                log_probs=self.log_probs, 
                dones=self.dones
            )
            self.clear_memory()


def train(optimizers=['adam']):
    plot_scores_dict = {opt: [] for opt in optimizers}
    plot_mean_scores_dict = {opt: [] for opt in optimizers}
    plot_record_scores_dict = {opt: [] for opt in optimizers}  # New list to store record values
    total_score_dict = {opt: 0 for opt in optimizers}
    record_dict = {opt: 0 for opt in optimizers}

    # Ensure the directory for saving plots exists
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    for opt_name in optimizers:
        agent = Agent(optimizer_name=opt_name)
        game = SnakeGameAI()
        total_score = 0
        
        plt.figure()
        plt.ion()  # Turn on interactive mode for live updates
        plot_scores_line, = plt.plot(plot_scores_dict[opt_name], label=f'Scores ({opt_name})')
        plot_mean_scores_line, = plt.plot(plot_mean_scores_dict[opt_name], label=f'Mean Scores ({opt_name})')
        plot_record_scores_line, = plt.plot(plot_record_scores_dict[opt_name], label=f'Record ({opt_name})', linestyle='--')  # New line for the record
        plt.xlabel('Games')
        plt.ylabel('Scores')
        plt.title(f'Performance with PPO Algorithm')
        plt.legend()

        while True:
            state_old = agent.get_state(game)
            action, log_prob = agent.get_action(state_old)
            reward, done, score = game.play_step(action)

            state_new = agent.get_state(game)
            agent.store_experience(state_old, action, reward, log_prob, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train()

                # Update scores
                if score > record_dict[opt_name]:
                    record_dict[opt_name] = score
                    # Save the model (if needed)
                    # agent.model.save(file_name=f'model_{opt_name}.pth') 

                # Log scores
                print(f'Optimizer: {opt_name}, Game {agent.n_games}, Score {score}, Record: {record_dict[opt_name]}')

                # Update plot data
                plot_scores_dict[opt_name].append(score)
                total_score_dict[opt_name] += score
                mean_score = total_score_dict[opt_name] / agent.n_games
                plot_mean_scores_dict[opt_name].append(mean_score)
                plot_record_scores_dict[opt_name].append(record_dict[opt_name])  # Append the new record value

                # Update the plot lines
                plot_scores_line.set_ydata(plot_scores_dict[opt_name])
                plot_mean_scores_line.set_ydata(plot_mean_scores_dict[opt_name])
                plot_record_scores_line.set_ydata(plot_record_scores_dict[opt_name])  # Update record plot
                plot_scores_line.set_xdata(range(len(plot_scores_dict[opt_name])))
                plot_mean_scores_line.set_xdata(range(len(plot_mean_scores_dict[opt_name])))
                plot_record_scores_line.set_xdata(range(len(plot_record_scores_dict[opt_name])))  # Update record X-axis

                # Adjust the plot limits
                plt.ylim(0, max(plot_scores_dict[opt_name] + [mean_score] + [record_dict[opt_name]], default=1))  # Adjust for record
                plt.xlim(0, len(plot_scores_dict[opt_name]))
                
                # Draw and save the plot every 100 games
                if agent.n_games % 100 == 0:
                    plt.savefig(f'plots/plot_{opt_name}_{agent.n_games}.png')

                plt.draw()
                plt.pause(0.1)  # Pause to allow the plot to update

            if agent.n_games >= 300:
                break

if __name__ == '__main__':
    train()
