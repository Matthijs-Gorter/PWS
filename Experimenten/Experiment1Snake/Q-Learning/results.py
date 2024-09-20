import csv
import matplotlib.pyplot as plt
import seaborn as sns

games = []
scores = []

with open('games_scores.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header
    for row in reader:
        games.append(int(row[0]))
        scores.append(int(row[1]))


# Set seaborn style for a professional look
sns.set_theme(style="whitegrid")

# Create the figure and axis
plt.figure(figsize=(10, 6))

# Plot the line graph for the averaged scores
plt.plot(games, scores, linestyle='-', color='blue', label='Q_Learning', linewidth=0.5)
plt.savefig('results.png', dpi=500)  # Save as PNG with 300 DPI

# Adding labels and title
plt.xlabel("Spel", fontsize=14)
plt.ylabel("Score", fontsize=14)

# Display the plot
plt.tight_layout()
plt.legend()
plt.show()