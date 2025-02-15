import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from scipy.stats import kendalltau

# Data inladen
df = pd.read_csv('results.csv')

# Algemene opmaak voor wetenschappelijke plots
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (15,10)})

# 1. Prestatie over tijd (Score)
plt.figure()
plt.plot(df['Episode'], df['Score'], 'b-', alpha=0.3, label='Raw Score')
plt.plot(df['Episode'], df['Score'].rolling(window=10).mean(), 'r-', label='Moving Average (window=10)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN Learning Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 2. Epsilon decay en Score relatie
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Epsilon', color=color)
ax1.plot(df['Episode'], df['Epsilon'], color=color, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('ApplesEaten', color=color)
ax2.plot(df['Episode'], df['ApplesEaten'], color=color, alpha=0.3)
ax2.plot(df['Episode'], df['ApplesEaten'].rolling(window=50).mean(), color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Exploration-Exploitation Tradeoff (Epsilon vs Score)')
plt.grid(True)
plt.tight_layout()

# 3. Loss ontwikkeling (filter NaN waardes)
loss_df = df.dropna(subset=['Loss'])

plt.figure()
plt.plot(loss_df['Episode'], loss_df['Loss'], 'g-')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('DQN Training Loss Development')
plt.grid(True)
plt.tight_layout()

# 4. Apples verzameld over tijd
plt.figure()
plt.plot(df['Episode'], df['ApplesEaten'], 'g-', alpha=0.3, label='Raw Apples')
plt.plot(df['Episode'], df['ApplesEaten'].rolling(window=10).mean(), 'm-', label='Moving Average (window=10)')
plt.xlabel('Episode')
plt.ylabel('Apples Collected')
plt.title('Apples Collected Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 5. Samengestelde visualisatie met Apples
fig, axs = plt.subplots(5, 1, sharex=True, figsize=(15, 12))  # Adjusted for 5 subplots
metrics = ['Score', 'Epsilon', 'Loss', 'Steps', 'ApplesEaten']
colors = ['blue', 'red', 'green', 'purple', 'orange']

for i, metric in enumerate(metrics):
    if metric == 'Loss':
        axs[i].plot(loss_df['Episode'], loss_df[metric], color=colors[i])
    else:
        axs[i].plot(df['Episode'], df[metric], color=colors[i], alpha=0.3)
        if metric == 'Score':  # Voeg moving average alleen toe voor Score
            axs[i].plot(df['Episode'], df[metric].rolling(window=10).mean(), color=colors[i])
    axs[i].set_ylabel(metric)
    axs[i].grid(True)

axs[-1].set_xlabel('Episode')
plt.suptitle('DQN Training Metrics Development')
plt.tight_layout()

# Basis statistieken
print(df.describe())

# Correlatie matrix inclusief Apples
print(df.corr())

# Performancetrend test (Mann-Kendall) voor Score en Apples
tau_score, p_value_score = kendalltau(df['Episode'], df['Score'])
tau_apples, p_value_apples = kendalltau(df['Episode'], df['ApplesEaten'])
print(f"Kendall's tau voor Score: {tau_score:.3f}, p-waarde: {p_value_score:.4f}")
print(f"Kendall's tau voor Apples: {tau_apples:.3f}, p-waarde: {p_value_apples:.4f}")

# Opslaan van alle plots
plt.savefig('dqn_analysis.png', dpi=300, bbox_inches='tight')

# Toon alle plots
plt.show()