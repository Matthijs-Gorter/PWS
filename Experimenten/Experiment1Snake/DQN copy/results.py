import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


games = []
scores = []

# Read data with pandas for efficiency
data = pd.read_csv('games_scores.csv')

# Extract data
games = data['Game'].values  # Assuming 'Game' is the column name
scores = data['Score'].values  # Assuming 'Score' is the column name

# Optional: Smoothing the data using a rolling average
data['Smoothed_Score'] = data['Score'].rolling(window=50).mean()

# Set seaborn style for a professional look
sns.set_theme(style="whitegrid")

# Create the figure and axis
plt.figure(figsize=(10, 6))

# Plot the line graph for the averaged scores
plt.plot(games, scores, linestyle='-', color='blue', label='Q_Learning', linewidth=0.5)
plt.plot(games, data['Smoothed_Score'], linestyle='-', color='red', label='Smoothed Scores', linewidth=2)

# Adding labels and title
plt.xlabel("Spel", fontsize=14)
plt.ylabel("Score", fontsize=14)

# Display the plot
plt.tight_layout()
plt.legend()
plt.savefig('results.png', dpi=500)
plt.show()