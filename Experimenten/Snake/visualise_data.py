import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Configuration constants
DATA_PATHS = {
    'DQN': Path('./DQN/results.csv'),
    'PPO': Path('./PPO/results.csv'),
    'Q-Learning': Path('./Q-Learning/results.csv')
}
GRAPH_DIR = Path('./graphs')
WINDOW_SIZE = 100  # Smoothing window for 15,000 time steps (empirically chosen)

def load_data(algorithm):
    """
    Load CSV data for specified algorithm.
    
    Args:
        algorithm: One of 'DQN', 'PPO', or 'Q-Learning'
    
    Returns:
        pandas.DataFrame: Loaded and validated data
    """
    path = DATA_PATHS.get(algorithm)
    if not path or not path.exists():
        raise FileNotFoundError(f"Data file missing for {algorithm} at {path}")
    
    df = pd.read_csv(path)
    
    # Basic data validation
    required_columns = {
        'DQN': ['Episode', 'TotalReward', 'ApplesEaten', 'AvgLoss', 'Epsilon', 'StepsPerApple', 'TotalTime'],
        'PPO': ['Episode', 'TotalReward', 'ApplesEaten', 'Loss', 'Entropy', 'StepsPerApple', 'TotalTime'],
        'Q-Learning': ['Episode', 'TotalReward', 'ApplesEaten', 'AvgLoss', 'Epsilon', 'StepsPerApple', 'TotalTime']
    }
    missing = set(required_columns[algorithm]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {algorithm} data: {missing}")
    
    return df

def preprocess_data(df, algorithm, window_size=WINDOW_SIZE):
    """
    Clean data and compute moving averages for relevant metrics.
    
    Args:
        df: Raw dataframe from load_data
        algorithm: Algorithm identifier
        window_size: Size of moving average window
    
    Returns:
        pandas.DataFrame: Processed dataframe with moving averages
    """
    # Convert numeric types and handle missing values
    numeric_cols = df.columns.drop('Episode') if 'Episode' in df.columns else df.columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    # Compute moving averages for relevant metrics
    ma_columns = {
        'DQN': ['TotalReward', 'ApplesEaten', 'AvgLoss', 'StepsPerApple', 'Epsilon'],
        'PPO': ['TotalReward', 'ApplesEaten', 'Loss', 'Entropy', 'StepsPerApple'],
        'Q-Learning': ['TotalReward', 'ApplesEaten', 'AvgLoss', 'StepsPerApple', 'Epsilon']
    }
    
    for col in ma_columns[algorithm]:
        df[f'{col}_MA'] = df[col].rolling(window_size, min_periods=1).mean()
    
    return df

def plot_performance(algorithm, df):
    """Generate performance visualization (TotalReward and ApplesEaten)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Episode-based plot
    ax1.plot(df['Episode'], df['TotalReward_MA'], label='Total Reward (MA)', color='tab:blue')
    ax1.set_xlabel('Episode', fontsize=10)
    ax1.set_ylabel('Total Reward', color='tab:blue', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax1b = ax1.twinx()
    ax1b.plot(df['Episode'], df['ApplesEaten_MA'], label='Apples Eaten (MA)', color='tab:orange')
    ax1b.set_ylabel('Apples Eaten', color='tab:orange', fontsize=10)
    ax1b.tick_params(axis='y', labelcolor='tab:orange')
    
    ax1.set_title(f'{algorithm} Performance Metrics by Episode', fontsize=12)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
    
    # Time-based plot
    ax2.plot(df['TotalTime'], df['TotalReward_MA'], label='Total Reward (MA)', color='tab:blue')
    ax2.set_xlabel('Training Time (seconds)', fontsize=10)
    ax2.set_ylabel('Total Reward', color='tab:blue', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2b = ax2.twinx()
    ax2b.plot(df['TotalTime'], df['ApplesEaten_MA'], label='Apples Eaten (MA)', color='tab:orange')
    ax2b.set_ylabel('Apples Eaten', color='tab:orange', fontsize=10)
    ax2b.tick_params(axis='y', labelcolor='tab:orange')
    
    ax2.set_title(f'{algorithm} Performance Metrics by Training Time', fontsize=12)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f'{algorithm}_performance.png', bbox_inches='tight')
    plt.close()

def plot_learning_process(algorithm, df):
    """Generate learning process visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if algorithm == 'PPO':
        ax.plot(df['Episode'], df['Loss_MA'], label='Loss (MA)', color='tab:red')
        ax.set_ylabel('Policy Loss', color='tab:red', fontsize=10)
        ax.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax.twinx()
        ax2.plot(df['Episode'], df['Entropy_MA'], label='Entropy (MA)', color='tab:green')
        ax2.set_ylabel('Policy Entropy', color='tab:green', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='tab:green')
    else:
        ax.plot(df['Episode'], df['AvgLoss_MA'], label='Average Loss (MA)', color='tab:red')
        ax.set_ylabel('Q-Loss', color='tab:red', fontsize=10)
        ax.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax.twinx()
        ax2.plot(df['Episode'], df['Epsilon_MA'], label='Epsilon (MA)', color='tab:purple')
        ax2.set_ylabel('Exploration Rate', color='tab:purple', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='tab:purple')
    
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_title(f'{algorithm} Learning Process Metrics', fontsize=12)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f'{algorithm}_learning_process.png', bbox_inches='tight')
    plt.close()

def plot_efficiency(algorithm, df):
    """Generate efficiency visualization (StepsPerApple)"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Episode'], df['StepsPerApple_MA'], 
             label='Moving Average', color='tab:blue')
    plt.scatter(df['Episode'], df['StepsPerApple'], 
                label='Raw Values', color='tab:blue', alpha=0.3, s=10)
    
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Steps per Apple', fontsize=10)
    plt.title(f'{algorithm} Efficiency Development', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f'{algorithm}_efficiency.png', bbox_inches='tight')
    plt.close()

def plot_comparative_analysis(dqn_df, ppo_df, ql_df):
    """Generate comparative visualizations"""
    # Learning Speed Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_df['Episode'], dqn_df['TotalReward_MA'], label='DQN')
    plt.plot(ppo_df['Episode'], ppo_df['TotalReward_MA'], label='PPO')
    plt.plot(ql_df['Episode'], ql_df['TotalReward_MA'], label='Q-Learning')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Smoothed Total Reward', fontsize=10)
    plt.title('Comparative Learning Speed (Smoothed Total Reward)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'comparative_learning_speed.png', bbox_inches='tight')
    plt.close()
    
    # Final Performance Comparison
    metrics = ['TotalReward', 'ApplesEaten']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for ax, metric in zip(axes, metrics):
        dqn_mean = dqn_df[metric].iloc[-WINDOW_SIZE:].mean()
        ppo_mean = ppo_df[metric].iloc[-WINDOW_SIZE:].mean()
        ql_mean = ql_df[metric].iloc[-WINDOW_SIZE:].mean()
        
        ax.bar(['DQN', 'PPO', 'Q-Learning'], [dqn_mean, ppo_mean, ql_mean],
               color=['tab:blue', 'tab:orange', 'tab:green'])
        ax.set_title(f'Final {metric} Comparison', fontsize=12)
        ax.set_ylabel(f'Average {metric} (Last {WINDOW_SIZE} Episodes)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'comparative_final_performance.png', bbox_inches='tight')
    plt.close()

def main():
    """Main analysis workflow"""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    algorithms = ['DQN', 'PPO', 'Q-Learning']
    data = {}
    for algo in algorithms:
        raw_df = load_data(algo)
        processed_df = preprocess_data(raw_df, algo)
        data[algo] = processed_df
    
    # Generate individual algorithm visualizations
    for algo in algorithms:
        plot_performance(algo, data[algo])
        plot_learning_process(algo, data[algo])
        plot_efficiency(algo, data[algo])
    
    # Generate comparative visualizations
    plot_comparative_analysis(data['DQN'], data['PPO'], data['Q-Learning'])

if __name__ == '__main__':
    main()