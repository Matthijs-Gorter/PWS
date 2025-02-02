import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import csv
import os

# Environment configuration
GRID_SIZE = 10
INPUT_SIZE = 9   # Direction (4) + food position (2) + dangers (3)
OUTPUT_SIZE = 3  # Actions: left, straight, right

# Hyperparameter ranges for tuning
HYPERPARAM_RANGES = {
    'lr': (1e-4, 1e-2),  # log scale
    'gamma': (0.9, 0.999),
    'hidden_size': [64, 128, 256],
    'batch_size': [32, 64, 128],
    'replay_capacity': [5000, 10000, 20000],
    'epsilon_decay': (0.99, 0.9999),
    'target_update': [5, 10, 20]
}

class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # Initial direction: right
        self.food = self._spawn_food()
        self.done = False
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
        
        # Direction (one-hot encoding)
        dir_vec = [0] * 4
        if self.direction == (0, 1):   dir_vec[1] = 1  # right
        elif self.direction == (0, -1): dir_vec[3] = 1  # left
        elif self.direction == (1, 0): dir_vec[2] = 1  # down
        elif self.direction == (-1, 0): dir_vec[0] = 1  # up
        
        # Danger detection
        dangers = []
        current_dir = self.direction
        for rel_dir in [(-current_dir[1], current_dir[0]),  # left
                        current_dir,                        # straight
                        (current_dir[1], -current_dir[0])]: # right
            new_head = (head[0] + rel_dir[0], head[1] + rel_dir[1])
            danger = 0
            if (new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size or
                new_head in self.snake):
                danger = 1
            dangers.append(danger)
        
        return np.array(dir_vec + [dx, dy] + dangers, dtype=np.float32)

    def step(self, action):
        if action == 0:   # Turn left
            new_dir = (-self.direction[1], self.direction[0])
        elif action == 2: # Turn right
            new_dir = (self.direction[1], -self.direction[0])
        else:             # Straight
            new_dir = self.direction
        
        self.direction = new_dir
        new_head = (self.snake[0][0] + new_dir[0],
                    self.snake[0][1] + new_dir[1])
        
        # Collision check
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.done = True
            reward = -10
            return self.get_state(), reward, self.done, {}
        
        # Move snake
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self._spawn_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1
        
        self.done = False
        return self.get_state(), reward, self.done, {}

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, lr, gamma, batch_size, replay_capacity, epsilon_start, epsilon_end, epsilon_decay, target_update):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.steps = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, OUTPUT_SIZE-1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state).argmax().item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions)) 
        
        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def save_results_to_csv(results, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Score", "Epsilon", "Loss", "Steps"])
        writer.writerow(results)

def save_hyperparameters_to_csv(trial, hyperparams, filename="trial_parameters.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Trial", "Learning Rate", "Gamma", "Hidden Size", "Batch Size", "Replay Capacity", "Epsilon Decay", "Target Update"])
        writer.writerow([trial] + list(hyperparams.values()))

def sample_hyperparameters():
    params = {}
    params['lr'] = 10 ** np.random.uniform(np.log10(HYPERPARAM_RANGES['lr'][0]), np.log10(HYPERPARAM_RANGES['lr'][1]))
    params['gamma'] = np.random.uniform(*HYPERPARAM_RANGES['gamma'])
    params['hidden_size'] = random.choice(HYPERPARAM_RANGES['hidden_size'])
    params['batch_size'] = random.choice(HYPERPARAM_RANGES['batch_size'])
    params['replay_capacity'] = random.choice(HYPERPARAM_RANGES['replay_capacity'])
    params['epsilon_decay'] = np.random.uniform(*HYPERPARAM_RANGES['epsilon_decay'])
    params['target_update'] = random.choice(HYPERPARAM_RANGES['target_update'])
    params['epsilon_start'] = 0.9
    params['epsilon_end'] = 0.00
    return params

if __name__ == "__main__":
    num_trials = 10
    for trial in range(num_trials):
        hyperparams = sample_hyperparameters()
        save_hyperparameters_to_csv(trial, hyperparams)
        
        filename = f"trial_{trial}_results.csv"
        
        env = SnakeEnv(GRID_SIZE)
        agent = DQNAgent(
            input_size=INPUT_SIZE,
            hidden_size=hyperparams['hidden_size'],
            output_size=OUTPUT_SIZE,
            lr=hyperparams['lr'],
            gamma=hyperparams['gamma'],
            batch_size=hyperparams['batch_size'],
            replay_capacity=hyperparams['replay_capacity'],
            epsilon_start=hyperparams['epsilon_start'],
            epsilon_end=hyperparams['epsilon_end'],
            epsilon_decay=hyperparams['epsilon_decay'],
            target_update=hyperparams['target_update']
        )
        
        scores = []
        for episode in range(100):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            loss = None
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                current_loss = agent.train()
                if current_loss is not None:
                    loss = current_loss
                steps += 1
                if steps >= 1000: break
            
            scores.append(total_reward)
            
            if episode % agent.target_update == 0:
                agent.update_target()
            
            save_results_to_csv(
                [episode, total_reward, agent.epsilon, loss if loss is not None else "N/A", steps],
                filename
            )
            
            print(f"Trial {trial} | Episode {episode:3d} | Score: {total_reward:6.1f} | Epsilon: {agent.epsilon:.2f} | Loss: {loss if loss is not None else 'N/A'} | Steps: {steps}")