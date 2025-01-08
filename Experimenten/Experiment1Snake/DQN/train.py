import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch_directml
import random
import time
import csv

# Game settings
WINDOW_WIDTH, WINDOW_HEIGHT = 80, 60  # Reduced size for pixel representation
GRID_SIZE = 2  # Reduced grid size to match new window dimensions
CHANNELS = 6  # 1 for grayscale, 4 for direction

# Directions
DIRECTIONS = {"UP": (0, -GRID_SIZE), "DOWN": (0, GRID_SIZE), 
             "LEFT": (-GRID_SIZE, 0), "RIGHT": (GRID_SIZE, 0)}
LEFT_TURN = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
RIGHT_TURN = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}

# Deep Q-Learning settings
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.97
EPSILON = 1
EPSILON_DECAY = 0.95
NUM_ACTIONS = 3
REPLAY_MEMORY = deque(maxlen=100000)
BATCH_SIZE = 64

class DQNetwork(nn.Module):
    def __init__(self, action_size):
        super(DQNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(CHANNELS, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        self.conv_output_size = self._get_conv_output()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_size)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def _get_conv_output(self):
        # Helper function to calculate conv output size
        x = torch.zeros(1, CHANNELS, WINDOW_HEIGHT, WINDOW_WIDTH)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return int(np.prod(x.size()))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SnakeGame:
    def __init__(self):
        # Initialize the device first
        try:
            self.device = torch_directml.device()
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize DQNetwork
        self.dqn = DQNetwork(NUM_ACTIONS).to(self.device)
        self.target_dqn = DQNetwork(NUM_ACTIONS).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Rest of the initialization
        print(f"Is DirectML available: {self.device is not None}")
        self.optimizer = torch.optim.SGD(self.dqn.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=100)
        self.criterion = nn.MSELoss()

        self.epsilon = EPSILON
        self.min_epsilon = 0.01
        self.epsilon_decay = EPSILON_DECAY

        self.game_history = []
        self.score_history = []
        self.reward_history = []
        self.loss_history = []

    def get_state(self, snake_body, fruit_pos, direction):
        """
        Construct the game state as a tensor with separate channels for:
        - Snake's body
        - Fruit's position
        - Current direction (encoded as one-hot across the entire grid)
        """
        # Initialize a state tensor with separate channels
        state = np.zeros((CHANNELS, WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.float32)
        
        # Channel 0: Snake body (value = 1.0)
        for segment in snake_body:
            x, y = segment[0] // GRID_SIZE, segment[1] // GRID_SIZE
            if 0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT:
                state[0, y, x] = 1.0
        
        # Channel 1: Fruit position (value = 1.0)
        fruit_x, fruit_y = fruit_pos[0] // GRID_SIZE, fruit_pos[1] // GRID_SIZE
        if 0 <= fruit_x < WINDOW_WIDTH and 0 <= fruit_y < WINDOW_HEIGHT:
            state[1, fruit_y, fruit_x] = 1.0
        
        # Channel 2-5: Direction encoding (one-hot)
        direction_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        dir_channel = direction_map[direction]
        state[2 + dir_channel, :, :] = 1.0  # Set the corresponding direction channel to 1.0
        
        return state


    def initialize_game(self):
        snake_pos = [WINDOW_WIDTH//2 * GRID_SIZE, WINDOW_HEIGHT//2 * GRID_SIZE]
        snake_body = [[snake_pos[0], snake_pos[1]],
                     [snake_pos[0]-GRID_SIZE, snake_pos[1]],
                     [snake_pos[0]-2*GRID_SIZE, snake_pos[1]]]
        direction = "RIGHT"
        score = 0
        return snake_pos, snake_body, direction, score

    def generate_fruit(self, snake_body):
        while True:
            fruit_pos = [
                random.randrange(0, WINDOW_WIDTH//GRID_SIZE) * GRID_SIZE,
                random.randrange(0, WINDOW_HEIGHT//GRID_SIZE) * GRID_SIZE
            ]
            if tuple(fruit_pos) not in map(tuple, snake_body):
                break
        return fruit_pos, True


    def check_collision(self, position, snake_body):
        return (position[0] < 0 or position[0] >= WINDOW_WIDTH * GRID_SIZE or
                position[1] < 0 or position[1] >= WINDOW_HEIGHT * GRID_SIZE or
                tuple(position) in map(tuple, snake_body[:-1]))

    def update_direction(self, current_direction, action_index):
        """
        Updates the snake's direction based on the current direction and action taken.
        Actions:
        0 - Turn Left
        1 - Go Straight
        2 - Turn Right
        """
        actions = ["LEFT", "STRAIGHT", "RIGHT"]
        action = actions[action_index]
        if action == "LEFT":
            return LEFT_TURN[current_direction]
        elif action == "RIGHT":
            return RIGHT_TURN[current_direction]
        return current_direction  # "STRAIGHT" means no change in direction


    def move_snake(self, snake_pos, snake_body, fruit_pos, fruit_spawn, direction, score):
        # Calculate distance to fruit before moving
        distance_before = abs(snake_pos[0] - fruit_pos[0]) + abs(snake_pos[1] - fruit_pos[1])
        
        # Update snake position
        snake_pos[0] += DIRECTIONS[direction][0]
        snake_pos[1] += DIRECTIONS[direction][1]
        
        done = False
        reward = -0.1
        if snake_pos == fruit_pos:
            score += 10
            reward = 1  # Positive reward for eating a fruit
            fruit_spawn = False
        else:
            snake_body.pop()
        
        if not fruit_spawn:
            fruit_pos, fruit_spawn = self.generate_fruit(snake_body)
        
        if self.check_collision(snake_pos, snake_body):
            reward = -10  # Negative reward for collision
            done = True
        
        # Calculate distance to fruit after moving
        distance_after = abs(snake_pos[0] - fruit_pos[0]) + abs(snake_pos[1] - fruit_pos[1])
        
        # Give a small positive reward for reducing the distance to the fruit
        if not done and distance_after < distance_before:
            reward += 0.2
        
        snake_body.insert(0, list(snake_pos))
        return snake_pos, fruit_pos, fruit_spawn, score, done, reward

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.dqn(state_tensor)
            return torch.argmax(q_values).item()

    def train_dqn(self):
        if len(REPLAY_MEMORY) < BATCH_SIZE:
            return 0
        
        batch = random.sample(REPLAY_MEMORY, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_actions = self.dqn(next_states_tensor).max(1)[1].unsqueeze(1)
            next_q_values = self.target_dqn(next_states_tensor).gather(1, next_actions).squeeze(1)
            target_q_values = rewards_tensor + (DISCOUNT_FACTOR * next_q_values * (1 - dones_tensor))
        
        current_q_values = self.dqn(states_tensor).gather(1, actions_tensor).squeeze(1)
        loss = self.criterion(current_q_values, target_q_values)
        
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_replay_memory(self, state, action, reward, next_state, done):
        REPLAY_MEMORY.append((state, action, reward, next_state, done))
        if len(REPLAY_MEMORY) > BATCH_SIZE:
            self.train_dqn()

    def update_target_network(self):
        # Soft update
        tau = 0.001
        for target_param, local_param in zip(self.target_dqn.parameters(), 
                                           self.dqn.parameters()):
            target_param.data.copy_(tau * local_param.data + 
                                  (1.0 - tau) * target_param.data)

    def train(self, num_games):
        best_score = 0
        for game_index in range(num_games):
            # Unpack the tuple returned by play_game
            score, total_reward, total_steps = self.play_game(training=True)
            
            self.game_history.append(game_index)
            self.score_history.append(score)
            self.reward_history.append(total_reward)
            
            # Update target network periodically
            if game_index % 10 == 0:
                self.update_target_network()
            
            # Pass only the score to the scheduler
            self.scheduler.step(float(score))
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, 
                            self.epsilon * self.epsilon_decay)
            
            if score > best_score:
                best_score = score
                self.save_model("best_model")
            
            if game_index % 1 == 0:
                avg_score = np.mean(self.score_history[-100:])
                print(f"Game {game_index}, Score: {score}, Avg Score: {avg_score:.2f}, Epsilon: {self.epsilon:.3f}, Time: {(time.time() - start_time):.2f} seconds, Steps: {total_steps}, Total Reward:{total_reward}")

                
    def play_game(self, training=True, max_steps=1000):
        snake_pos, snake_body, direction, score = self.initialize_game()
        fruit_pos, fruit_spawn = self.generate_fruit(snake_body)
        
        total_reward = 0
        steps_without_fruit = 0
        
        for step in range(max_steps):
            current_state = self.get_state(snake_body, fruit_pos, direction)
            
            if training:
                action_index = self.choose_action(current_state)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                    action_index = self.dqn(state_tensor).argmax().item()
            
            direction = self.update_direction(direction, action_index)
            
            snake_pos, fruit_pos, fruit_spawn, score, done, reward = self.move_snake(
                snake_pos, snake_body, fruit_pos, fruit_spawn, direction, score
            )

                
            if snake_pos == fruit_pos:
                steps_without_fruit = 0
            else:
                steps_without_fruit += 1
                
        
            
            total_reward += reward
            
            if training:
                next_state = self.get_state(snake_body, fruit_pos, direction)
                self.update_replay_memory(current_state, action_index, reward, next_state, done)
                
                if len(REPLAY_MEMORY) >= BATCH_SIZE:
                    loss = self.train_dqn()
                    self.loss_history.append(loss)
            
            if done or steps_without_fruit > 200:
                break
        
        if training:
            return score, total_reward, steps_without_fruit
        return score

    def save_model(self, filepath):
            # Save model and optimizer state dictionaries separately
            model_state = self.dqn.state_dict()
            optimizer_state = self.optimizer.state_dict()
            torch.save(model_state, filepath + "_model.pth")
            torch.save(optimizer_state, filepath + "_optimizer.pth")
            
            # Save other parameters as numpy arrays
            np.savez(filepath + "_params.npz",
                epsilon=self.epsilon,
                game_history=np.array(self.game_history),
                score_history=np.array(self.score_history)
            )

    def load_model(self, filepath):
        # Load model state with weights_only=True
        model_state = torch.load(filepath + "_model.pth", weights_only=False)
        self.dqn.load_state_dict(model_state)
        
        # Load optimizer state with weights_only=True
        optimizer_state = torch.load(filepath + "_optimizer.pth", weights_only=True)
        self.optimizer.load_state_dict(optimizer_state)
        
        # Load other parameters from numpy file
        params = np.load(filepath + "_params.npz")
        self.epsilon = float(params['epsilon'])
        self.game_history = params['game_history'].tolist()
        self.score_history = params['score_history'].tolist()

    def save_results_to_csv(self, filename="games_scores.csv"):
            """Save game results to a CSV file."""
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Game', 'Score'])  # Write header
                for game, score in zip(self.game_history, self.score_history):
                    writer.writerow([game, score])
            print(f"Results saved to {filename}")

if __name__ == "__main__":
    game = SnakeGame()
    start_time = time.time()
    game.train(200)
    print(f"Training completed in {(time.time() - start_time):.2f} seconds")
    game.save_model("snake_model_pixel")
    game.save_results_to_csv()