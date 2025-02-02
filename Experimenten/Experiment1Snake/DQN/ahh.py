import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load hyperparameters and trial results
params_df = pd.read_csv("trial_parameters.csv")
all_trials_data = []

for trial in range(10):
    results_df = pd.read_csv(f"trial_{trial}_results.csv")
    trial_metrics = {
        "trial": trial,
        "mean_score": results_df["Score"].mean(),
        "max_score": results_df["Score"].max(),
        "last_100_avg": results_df["Score"].tail(100).mean(),
        "stability": results_df["Score"].std()
    }
    all_trials_data.append(trial_metrics)

metrics_df = pd.DataFrame(all_trials_data)
combined_df = pd.merge(params_df, metrics_df, on="trial")

# Calculate correlations
correlation_matrix = combined_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix[["mean_score", "max_score", "last_100_avg"]], annot=True)
plt.title("Correlation Between Hyperparameters and Performance")
plt.show()