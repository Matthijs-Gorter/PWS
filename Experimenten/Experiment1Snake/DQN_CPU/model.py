import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from PIL import Image
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat(state), 
                torch.tensor(action), 
                torch.tensor(reward), 
                torch.cat(next_state),
                torch.tensor(done))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, n_actions, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(100000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.target_update = 1000
        self.epsilon = self.eps_start
        
        self.steps_done = 0
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], 
                              device=self.device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones.float()))
        
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(-1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done += 1
        
        return loss.item()

    def preprocess_state(self, state):
        # Convert numpy array to PIL Image and resize
        state = Image.fromarray(state)
        state = state.resize((84, 84))
        # Convert to grayscale
        state = state.convert('L')
        # Convert to numpy array and normalize
        state = np.array(state) / 255.0
        # Add channel dimension and convert to tensor
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        return state.to(self.device)

    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])