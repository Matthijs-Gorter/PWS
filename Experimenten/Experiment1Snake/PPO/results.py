import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

# Data inladen
df = pd.read_csv('ppo_results.csv')

# Stijl instellen
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (15, 10),
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# 1. Leercurve met voortschrijdend gemiddelde
plt.figure(figsize=(12, 6))
plt.plot(df['Episode'], df['ApplesEaten'], 'b-', alpha=0.2, label='Aantal apples')
plt.plot(df['Episode'], df['ApplesEaten'].rolling(window=100).mean(), 'r-', lw=2, label='100-Episode Bewegend gemiddelde')
plt.xlabel('Episode Number')
plt.ylabel('Aantal gegeten apples')
plt.title('PPO')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=600)
plt.close()

# 2. Verliescurves en Entropie
fig, ax1 = plt.subplots(figsize=(12,6))

color = 'tab:red'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Loss', color=color)
ax1.plot(df['Episode'], df['Loss'], color=color, alpha=0.3)
ax1.plot(df['Episode'], df['Loss'].rolling(20).mean(), color=color, lw=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Entropy', color=color) 
ax2.plot(df['Episode'], df['Entropy'], color=color, alpha=0.3)
ax2.plot(df['Episode'], df['Entropy'].rolling(20).mean(), color=color, lw=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Metrics: Loss and Entropy')
plt.tight_layout()
plt.savefig('loss_entropy.png', dpi=600)
plt.close()

# 3. Appels gegeten vs Score
plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x='ApplesEaten', y='Score', alpha=0.6, edgecolor=None)
plt.plot(df['ApplesEaten'].rolling(50).mean(), 
         df['Score'].rolling(50).mean(), 
         'r--', lw=2, 
         label='Rolling Mean (window=50)')

# Kendall's Tau berekenen
tau, p_value = kendalltau(df['ApplesEaten'], df['Score'])
plt.text(0.05, 0.95, 
         f"Kendall's τ = {tau:.2f} (p = {p_value:.3f})", 
         transform=plt.gca().transAxes)

plt.xlabel('Apples Eaten')
plt.ylabel('Episode Score')
plt.title('Apples Eaten vs Total Score')
plt.legend()
plt.tight_layout()
plt.savefig('apples_vs_score.png', dpi=600)
plt.close()

# 4. Gecombineerde metrics in subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 12))

# Score
axs[0].plot(df['Episode'], df['Score'].rolling(20).mean(), 'b-')
axs[0].set_title('Score (20-Episode Moving Average)')

# Appels
axs[1].plot(df['Episode'], df['ApplesEaten'].rolling(20).mean(), 'g-')
axs[1].set_title('Apples Eaten (20-Episode Moving Average)')

# Entropie
axs[2].plot(df['Episode'], df['Entropy'].rolling(20).mean(), 'r-')
axs[2].set_title('Policy Entropy (20-Episode Moving Average)')

for ax in axs:
    ax.grid(True)
    ax.set_xlabel('Episode')

plt.tight_layout()
plt.savefig('combined_metrics.png', dpi=600)
plt.close()