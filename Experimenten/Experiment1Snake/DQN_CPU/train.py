from snake_env import SnakeEnv
from model import DQNAgent
import torch
import numpy as np
from tqdm import tqdm

def train():
    # Initialize environment and agent
    env = SnakeEnv()
    state_shape = env.get_state_shape()
    n_actions = env.get_action_space()
    agent = DQNAgent(state_shape, n_actions)
    
    # Training parameters
    n_episodes = 100
    max_steps = 1000
    
    # Training loop
    best_reward = -float('inf')
    
    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        state = agent.preprocess_state(state)
        episode_reward = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done = env.step(action.item())
            next_state = agent.preprocess_state(next_state)
            episode_reward += reward
            
            # Store transition and optimize model
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.optimize_model()
            
            state = next_state
            
            if done:
                break
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model('best_snake_model.pth')
            
        # Print progress every 100 episodes
        if episode % 1 == 0:
            print(f'Episode {episode}: Reward = {episode_reward}, Best = {best_reward}, Epsilon = {agent.epsilon:.3f}')

if __name__ == '__main__':
    train()