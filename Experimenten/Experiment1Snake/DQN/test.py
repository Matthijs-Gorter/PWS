import torch
import torch_directml
import numpy as np
import random
from collections import deque
from PIL import Image, ImageDraw

class SnakeEnv:
    def __init__(self, width=84, height=84):
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
        # Create a blank image
        img = Image.new('RGB', (self.width, self.height), 'black')
        draw = ImageDraw.Draw(img)
        
        # Draw snake
        for segment in self.snake:
            draw.point(segment, fill='white')
        
        # Draw food
        draw.point(self.food, fill='red')
        
        # Convert to numpy array and normalize
        state = np.array(img)
        state = state.transpose((2, 0, 1)) # Convert to PyTorch format (C, H, W)
        state = state / 255.0
        return state
    
    def step(self, action):
        self.steps += 1
        # Convert action (0-3) to direction
        if action == 0:   # UP
            self.direction = (0, -1)
        elif action == 1: # RIGHT
            self.direction = (1, 0)
        elif action == 2: # DOWN
            self.direction = (0, 1)
        elif action == 3: # LEFT
            self.direction = (-1, 0)
            
        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check if game over
        done = False
        reward = 0
        
        # Hit wall
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            done = True
            reward = -1
        else:
            self.snake.insert(0, new_head)
            
            # Eat food
            if new_head == self.food:
                self.score += 1
                reward = 1
                self.food = self._place_food()
            else:
                self.snake.pop()
                
            # Small negative reward for each step to encourage finding food quickly
            if not done:
                reward += -0.01
                
        return self._get_state(), reward, done, {"score": self.score}

class DQN(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
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
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states), 
                np.array(dones, dtype=np.uint8))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.buffer = ReplayBuffer(100000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 1000
        self.steps_done = 0
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def main():
    # Set up DirectML device
    device = torch_directml.device()
    
    # Initialize environment and agent
    env = SnakeEnv()
    state_shape = (3, 84, 84)  # RGB image
    n_actions = 4  # UP, RIGHT, DOWN, LEFT
    agent = DQNAgent(state_shape, n_actions, device)
    
    num_episodes = 10000
    max_steps = 1000
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        if episode % 1 == 0:
            print(f"Episode {episode}, Score: {info['score']}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    main()