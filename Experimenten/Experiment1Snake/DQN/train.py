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
GRID_SIZE = 10
INPUT_SIZE = 9   # Richting (4) + voedsel positie (2) + gevaren (3)
HIDDEN_SIZE = 128
OUTPUT_SIZE = 3  # Acties: links, rechtdoor, rechts

# DQN hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
REPLAY_CAPACITY = 10000
EPSILON_START = 0.9
EPSILON_END = 0.00
EPSILON_DECAY = 0.997
TARGET_UPDATE = 10

class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # Start richting: rechts
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
            reward = -10
            return self.get_state(), reward, self.done, {}
        
        # Voedsel verzamelen
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
    def __init__(self, input_size, hidden_size, output_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        self.epsilon = EPSILON_START
        self.memory = ReplayBuffer(REPLAY_CAPACITY)
        self.steps = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, OUTPUT_SIZE-1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state).argmax().item()
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions)) 
        
        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * GAMMA * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def save_results_to_csv(results, filename="results.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Score", "Epsilon", "Loss", "Steps"])
        writer.writerow(results)

if __name__ == "__main__":
    # Training loop
    env = SnakeEnv(GRID_SIZE)
    agent = DQNAgent(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    scores = []
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            loss = agent.train()
            steps += 1
        
        scores.append(total_reward)
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
        
        print(f"Episode {episode:3d} | Score: {total_reward:6.1f} | Epsilon: {agent.epsilon:.2f} | Loss: {loss:.4f} | Steps: {steps}" if loss is not None else 
              f"Episode {episode:3d} | Score: {total_reward:6.1f} | Epsilon: {agent.epsilon:.2f} | Loss: N/A | Steps: {steps}")
        save_results_to_csv([episode, total_reward, agent.epsilon, loss if loss is not None else "N/A", steps])
        
    torch.save(agent.policy_net.state_dict(), "dqn_snake.pth")