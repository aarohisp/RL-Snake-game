import matplotlib.pyplot as plt

# Read data from file
def read_data(filename):
    games = []
    scores = []
    records = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            if len(parts) == 3:
                game_num = int(parts[0].split()[1])
                score = int(parts[1].split()[1])
                record = int(parts[2].split()[1])
                
                games.append(game_num)
                scores.append(score)
                records.append(record)
                
    return games, scores, records

# Compute mean scores
def compute_mean_scores(scores):
    mean_scores = []
    total_score = 0
    for i, score in enumerate(scores):
        total_score += score
        mean_scores.append(total_score / (i + 1))
    return mean_scores

# Plot the data
def plot_data(games, scores, mean_scores, records):
    plt.figure()
    
    # Plot scores
    plt.plot(games, scores, label='Scores', color='blue')
    
    # Plot mean scores
    plt.plot(games, mean_scores, label='Mean Scores', color='green')
    
    # Plot records
    plt.plot(games, records, label='Records', color='red', linestyle='--')
    
    plt.xlabel('Games')
    plt.ylabel('Scores')
    plt.title('Game Scores and Records')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('plots/game_scores_plot.png')
    plt.show()

# Main function
def main():
    filename = 'game_scores.txt'
    
    games, scores, records = read_data(filename)
    mean_scores = compute_mean_scores(scores)
    plot_data(games, scores, mean_scores, records)

if __name__ == '__main__':
    main()
