import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import csv
import os

# Omgeving configuratie
GRID_SIZE = 20
INPUT_SIZE = 9
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

# PPO hyperparameters
LR = 0.0001
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.01
TRAJECTORY_LENGTH = 2048
MINIBATCH_SIZE = 64
PPO_EPOCHS = 4
MAX_GRAD_NORM = 0.5

class SnakeEnv:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # Start richting: rechts
        self.food = self._spawn_food()
        self.done = False
        self.apples_eaten = 0
        return self.get_state()

    def _spawn_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), 
                    random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def get_state(self):
        head = self.snake[0]
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]
        
        # Richting (one-hot encoding)
        dir_vec = [0] * 4
        if self.direction == (0, 1):   dir_vec[1] = 1  # rechts
        elif self.direction == (0, -1): dir_vec[3] = 1  # links
        elif self.direction == (1, 0): dir_vec[2] = 1  # onder
        elif self.direction == (-1, 0): dir_vec[0] = 1  # boven
        
        # Gevaren detectie
        dangers = []
        current_dir = self.direction
        for rel_dir in [(-current_dir[1], current_dir[0]),  # links
                        current_dir,                        # rechtdoor
                        (current_dir[1], -current_dir[0])]: # rechts
            new_head = (head[0] + rel_dir[0], head[1] + rel_dir[1])
            danger = 0
            if (new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size or
                new_head in self.snake):
                danger = 1
            dangers.append(danger)
        
        return np.array(dir_vec + [dx, dy] + dangers, dtype=np.float32)

    def step(self, action):
        # Actie verwerken
        if action == 0:   # Links draaien
            new_dir = (-self.direction[1], self.direction[0])
        elif action == 2: # Rechts draaien
            new_dir = (self.direction[1], -self.direction[0])
        else:             # Rechtdoor
            new_dir = self.direction
        
        self.direction = new_dir
        new_head = (self.snake[0][0] + new_dir[0],
                    self.snake[0][1] + new_dir[1])
        
        # Botsing detectie
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.done = True
            reward = -50
            return self.get_state(), reward, self.done, {}
        
        # Voedsel verzamelen
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self._spawn_food()
            reward = 10
            self.apples_eaten += 1
        else:
            self.snake.pop()
            reward = 0
        
        self.done = False
        return self.get_state(), reward, self.done, {}

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, output_size)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def prepare_batch(self):
        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.log_probs),
                np.array(self.values),
                np.array(self.rewards),
                np.array(self.dones))

class PPOAgent:
    def __init__(self, input_size, hidden_size, output_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.buffer = PPOBuffer()
        self.mse_loss = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()
    
    def compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_value = 0
            else:
                delta = rewards[t] + GAMMA * last_value - values[t]
            advantages[t] = delta + GAMMA * GAE_LAMBDA * last_advantage
            last_advantage = advantages[t]
            last_value = values[t]
        
        return advantages
    
    def update(self):
        states, actions, old_log_probs, old_values, rewards, dones = self.buffer.prepare_batch()
        
        # Initialize metrics tracking
        total_loss = 0.0
        total_entropy = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_minibatches = 0
                
        # Bereken advantages en returns
        advantages = self.compute_advantages(rewards, old_values, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Training loop
        for _ in range(PPO_EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                idx = indices[start:end]
                
                # Nieuwe waarden berekenen
                logits, values = self.model(states[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[idx])
                                
                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Ratio berekenen
                ratio = (new_log_probs - old_log_probs[idx]).exp()
                
                # Policy loss
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1-PPO_EPSILON, 1+PPO_EPSILON) * advantages[idx]

                
                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.mse_loss(values, returns[idx].unsqueeze(-1))
                loss = policy_loss + CRITIC_DISCOUNT * value_loss - ENTROPY_BETA * entropy
                
                # Accumulate metrics
                total_loss += loss.item()
                total_entropy += entropy.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                num_minibatches += 1

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        # Calculate averages
        avg_loss = total_loss / num_minibatches
        avg_entropy = total_entropy / num_minibatches
        self.buffer.clear()
        return avg_loss, avg_entropy

def save_results_to_csv(results, filename="ppo_results.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Score", "ApplesEaten", "Loss", "Entropy"])
        writer.writerow(results)

if __name__ == "__main__":
    env = SnakeEnv(GRID_SIZE)
    agent = PPOAgent(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
    episode = 0
    total_steps = 0
    
    while True:
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Collect trajectory
            for _ in range(TRAJECTORY_LENGTH):
                if done:
                    break
                
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.buffer.store(state, action, log_prob.item(), value.item(), reward, done)
                
                state = next_state
                episode_reward += reward
            
            # Update policy and get metrics
            avg_loss, avg_entropy = agent.update()
            total_steps += 1
        
        # Logging with actual metrics
        apples_eaten = env.apples_eaten
        print(f"Episode {episode} | Reward: {episode_reward} | Apples: {apples_eaten} | Loss: {avg_loss:.2f} | Entropy: {avg_entropy:.2f}")
        save_results_to_csv([episode, episode_reward, apples_eaten, avg_loss, avg_entropy])
        
        episode += 1
        if episode % 100 == 0:
            torch.save(agent.model.state_dict(), f"ppo_snake_{episode}.pth")
