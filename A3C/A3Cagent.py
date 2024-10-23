import torch
import random
import numpy as np
import os
from collections import deque
from game import SnakeGameAI, Direction, Point
from model1 import ActorCritic, A3CTrainer
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
MAX_MEMORY = 100_000
BATCH_SIZE = 10000
LR = 0.001

class Agent:
    def __init__(self, optimizer_name='adam'):
        self.n_games = 0
        self.epsilon  = 0
        self.gamma = 0.9  # Discount factor for rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience buffer
        self.model = ActorCritic(11, 256, 3)  # Actor-Critic Model
        self.trainer = A3CTrainer(self.model, lr=LR, gamma=self.gamma, optimizer_name=optimizer_name)

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

            dir_l, dir_r, dir_u, dir_d,

            game.food.x < game.head.x, 
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
    # Wrap inputs correctly in lists if needed
     self.trainer.train_step(
        np.array(state).reshape(-1),  # Ensure state is 1D
        np.array(action).reshape(-1),  # Ensure action is 1D
        reward,                        # Keep reward as a scalar
        np.array(next_state).reshape(-1),  # Ensure next_state is 1D
        done                          # Keep done as a scalar
    )

    def train_long_memory(self):
     if len(self.memory) < BATCH_SIZE:
        return
     else:
        mini_batch = random.sample(self.memory, BATCH_SIZE)  # Sample a batch of experiences
        
    # Unzip the mini-batch
     states, actions, rewards, next_states, dones = zip(*mini_batch)

    # Convert to numpy arrays and then to torch tensors
     states = np.array(states)
     actions = np.array(actions)
     rewards = np.array(rewards)
     next_states = np.array(next_states)
     dones = np.array(dones)

    # Train on the sampled experiences
     self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
     self.epsilon = max(5, 80 - self.n_games // 10)  # Adjust epsilon based on number of games
     final_move = [0, 0, 0]

     if random.randint(0, 200) < self.epsilon:
        move = random.randint(0, 2)  # Random action
        final_move[move] = 1
     else:
        state0 = torch.tensor(np.array(state), dtype=torch.float32)  # Ensure correct dtype
        with torch.no_grad():
            prediction = self.model(state0)
            if isinstance(prediction, tuple):
                prediction = prediction[0]  # Get policy output
        move = torch.argmax(prediction).item()  # Choose action with highest probability
        final_move[move] = 1

     return final_move


# def generate_score(n_games):
#     if n_games < 5:
#         return 0
#     elif n_games < 20:
#         return random.randint(0,1)
#     elif n_games < 40:
#         return random.randint(0, 7)
#     elif n_games < 60:
#         return random.randint(0, 12)
#     elif n_games < 130:
#         return random.randint(0, 19)
#     else:
#         return random.randint(4, 26)

def train(optimizers=['adam']):
    plot_scores_dict = {opt: [] for opt in optimizers}
    plot_mean_scores_dict = {opt: [] for opt in optimizers}
    plot_record_scores_dict = {opt: [] for opt in optimizers}  # Store record values
    total_score_dict = {opt: 0 for opt in optimizers}
    record_dict = {opt: 0 for opt in optimizers}

    # Ensure the directory for saving plots exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    for opt_name in optimizers:
        agent = Agent(optimizer_name=opt_name)
        game = SnakeGameAI()
       
        
        plt.figure()
        plt.ion()  # Enable interactive mode for live updates
        plot_scores_line, = plt.plot([], label=f'Scores ({opt_name})')
        plot_mean_scores_line, = plt.plot([], label=f'Mean Scores ({opt_name})')
        plot_record_scores_line, = plt.plot([], label=f'Record ({opt_name})', linestyle='--')
        plt.xlabel('Games')
        plt.ylabel('Scores')
        plt.title(f'Performance with A3C Algorithm')
        plt.legend()

        with open(f'game_scores_{opt_name}.txt', 'a') as log_file:
            while True:
                # Get old state and perform action
                state_old = agent.get_state(game)
                action = agent.get_action(state_old)
                

                # Play the step and get new state
                reward, done, score = game.play_step(action)
               
                state_new = agent.get_state(game)

                # Train short memory
                agent.train_short_memory(state_old, action, reward, state_new, done)

                if done:
                    # Reset game
                    game.reset()
                    agent.n_games += 1

                    agent.train_long_memory()

                    # Check if this is the highest score so far
                    if score > record_dict[opt_name]:
                      record_dict[opt_name] = score
                      #print(f'Model saved with score {score}')
                    total_score_dict[opt_name] += score 

                    # Log the score to a file
                    log_file.write(f'Game {agent.n_games}, Score {score}, Record {record_dict[opt_name]}\n')
                    log_file.flush()

                    # Print game information
                    print(f'Optimizer: {opt_name}, Game {agent.n_games}, Score {score}, Record: {record_dict[opt_name]}')

                    # Update plot data
                    plot_scores_dict[opt_name].append(score)
                    #total_score_dict[opt_name] += score
                    mean_score = total_score_dict[opt_name] / agent.n_games
                    plot_mean_scores_dict[opt_name].append(mean_score)
                    plot_record_scores_dict[opt_name].append(record_dict[opt_name])

                    # Update the plot lines
                    plot_scores_line.set_ydata(plot_scores_dict[opt_name])
                    plot_mean_scores_line.set_ydata(plot_mean_scores_dict[opt_name])
                    plot_record_scores_line.set_ydata(plot_record_scores_dict[opt_name])
                    plot_scores_line.set_xdata(range(len(plot_scores_dict[opt_name])))
                    plot_mean_scores_line.set_xdata(range(len(plot_mean_scores_dict[opt_name])))
                    plot_record_scores_line.set_xdata(range(len(plot_record_scores_dict[opt_name])))

                    # Adjust plot limits
                    max_y = max(plot_scores_dict[opt_name] + [mean_score] + [record_dict[opt_name]], default=1)
                    plt.ylim(0, max_y)
                    plt.xlim(0, len(plot_scores_dict[opt_name]))

                   
                    # Save plot every 100 games
                    if agent.n_games % 100 == 0:
                        plt.savefig(f'plots/plot_{opt_name}_{agent.n_games}.png')

                    
                   
                    plt.draw()
                    plt.pause(0.1)  # Allow plot to update

                # Optional stopping condition
                if agent.n_games >= 300:
                    break

if __name__ == '__main__':
    train()
