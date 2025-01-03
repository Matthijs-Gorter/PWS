import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import time

# Configure PyTorch to use ROCm backend for AMD GPU
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SnakeGame:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        return self._get_state()
    
    def _place_food(self):
        while True:
            food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        state = np.zeros((4, self.width, self.height))
        
        # Snake head channel
        state[0, self.snake[0][0], self.snake[0][1]] = 1
        
        # Snake body channel
        for segment in self.snake[1:]:
            state[1, segment[0], segment[1]] = 1
        
        # Food channel
        state[2, self.food[0], self.food[1]] = 1
        
        # Direction channel
        state[3, self.snake[0][0], self.snake[0][1]] = (
            self.direction[0] + 1 if self.direction[0] != 0 else self.direction[1] + 2
        )
        
        return state
    
    def step(self, action):
        # Convert action (0,1,2,3) to direction
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        new_direction = directions[action]
        
        # Prevent 180-degree turns
        if (new_direction[0] != -self.direction[0] or new_direction[0] == 0) and \
           (new_direction[1] != -self.direction[1] or new_direction[1] == 0):
            self.direction = new_direction
        
        # Move snake
        new_head = (
            (self.snake[0][0] + self.direction[0]) % self.width,
            (self.snake[0][1] + self.direction[1]) % self.height
        )
        
        # Check collision with self
        if new_head in self.snake:
            return self._get_state(), -10, True
        
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
        
        self.steps += 1
        
        # End game if too many steps without eating
        done = self.steps > 100 * len(self.snake)
        
        return self._get_state(), reward, done

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, state_shape, n_actions):
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.00025)
        self.memory = deque(maxlen=100000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.target_update = 1000
        self.steps_done = 0
        
    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)
    
    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
        action_batch = torch.LongTensor(batch[1]).to(device)
        reward_batch = torch.FloatTensor(batch[2]).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def train():
    env = SnakeGame()
    state_shape = (4, env.width, env.height)
    n_actions = 4
    agent = DQNAgent(state_shape, n_actions)
    
    episodes = 10000
    best_score = 0
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            agent.store_transition(state, action, reward, next_state)
            agent.optimize_model()
            
            state = next_state
            
            if done:
                scores.append(env.score)
                if env.score > best_score:
                    best_score = env.score
                    torch.save(agent.policy_net.state_dict(), 'best_snake_model.pth')
                
                if episode % 1 == 0:
                    avg_score = np.mean(scores[-100:])
                    print(f"Episode {episode}, Average Score: {avg_score:.2f}, Best Score: {best_score}")
                break

if __name__ == "__main__":
    train()