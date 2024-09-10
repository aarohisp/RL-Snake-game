import torch
import random
import numpy as np
import os
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
MAX_MEMORY = 100_000
BATCH_SIZE = 10000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


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
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)

            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    # Ensure the directory for saving plots exists
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Initialize the plot
    plt.figure()
    plt.ion()  # Turn on interactive mode for live updates
    plot_scores_line, = plt.plot(plot_scores, label='Scores')
    plot_mean_scores_line, = plt.plot(plot_mean_scores, label='Mean Scores')
    plt.xlabel('Games')
    plt.ylabel('Scores')
    plt.title('Game Scores and Mean Scores')
    plt.legend()
    
    # Open a file for logging
    with open('game_scores.txt', 'a') as log_file:
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                # Log the game score and record to a file
                log_file.write(f'Game {agent.n_games}, Score {score}, Record {record}\n')
                log_file.flush()  # Ensure the log is written to the file immediately

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                # Update the plot data
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                # Update the plot lines
                plot_scores_line.set_ydata(plot_scores)
                plot_mean_scores_line.set_ydata(plot_mean_scores)
                plot_scores_line.set_xdata(range(len(plot_scores)))
                plot_mean_scores_line.set_xdata(range(len(plot_mean_scores)))

                # Adjust the plot limits
                plt.ylim(0, max(plot_scores + [mean_score], default=1))
                plt.xlim(0, len(plot_scores))

                # Draw and save the plot every 5 games
                if agent.n_games % 5 == 0:
                    plt.savefig(f'plots/plot_{agent.n_games}.png')
                
                plt.draw()
                plt.pause(0.1)  # Pause to allow the plot to update

    # Save the final plot when training ends
    plt.ioff()  # Turn off interactive mode
    plt.savefig(f'plots/plot_{agent.n_games}.png')
    plt.close()



if __name__ == '__main__':
    train()
