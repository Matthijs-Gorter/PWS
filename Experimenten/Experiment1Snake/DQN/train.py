import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import csv

# Game settings
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
GRID_SIZE = 20

# Directions
DIRECTIONS = {"UP": (0, -20), "DOWN": (0, 20), "LEFT": (-20, 0), "RIGHT": (20, 0)}
LEFT_TURN = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
RIGHT_TURN = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}

# Deep Q-Learning settings
LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0.97
EPSILON = 1
EPSILON_DECAY = 0.995
NUM_ACTIONS = 3
NUM_STATES = 6
REPLAY_MEMORY = deque(maxlen=50000)
BATCH_SIZE = 256

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SnakeGame:
    def __init__(self):
        self.dqn = DQNetwork(NUM_STATES, NUM_ACTIONS)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = EPSILON
        self.game_history = []
        self.score_history = []

    def initialize_game(self):
        snake_pos = [300, 300]
        snake_body = [[300, 300], [280, 300], [260, 300], [240, 300]]
        direction = "RIGHT"
        score = 0
        return snake_pos, snake_body, direction, score

    def generate_fruit(self, snake_body):
        while True:
            fruit_pos = [
                random.randrange(1, (WINDOW_WIDTH // GRID_SIZE)) * GRID_SIZE,
                random.randrange(1, (WINDOW_HEIGHT // GRID_SIZE)) * GRID_SIZE
            ]
            if tuple(fruit_pos) not in map(tuple, snake_body):
                break
        return fruit_pos, True

    def get_state(self, snake_pos, fruit_pos, snake_body, direction):
        direction_vector = DIRECTIONS[direction]
        left_vector = DIRECTIONS[LEFT_TURN[direction]]
        right_vector = DIRECTIONS[RIGHT_TURN[direction]]
        
        food_direction = (fruit_pos[0] - snake_pos[0], fruit_pos[1] - snake_pos[1])
        
        state = [
            self.check_collision([snake_pos[0] + direction_vector[0], snake_pos[1] + direction_vector[1]], snake_body),
            self.check_collision([snake_pos[0] + right_vector[0], snake_pos[1] + right_vector[1]], snake_body),
            self.check_collision([snake_pos[0] + left_vector[0], snake_pos[1] + left_vector[1]], snake_body),
            int(np.sign(direction_vector[0]) == np.sign(food_direction[0]) or 
                np.sign(direction_vector[1]) == np.sign(food_direction[1])),
            int(np.sign(right_vector[0]) == np.sign(food_direction[0]) or 
                np.sign(right_vector[1]) == np.sign(food_direction[1])),
            int(np.sign(left_vector[0]) == np.sign(food_direction[0]) or 
                np.sign(left_vector[1]) == np.sign(food_direction[1])),
        ]
        return np.array(state, dtype=np.float32)

    def check_collision(self, position, snake_body):
        return (position[0] < 0 or position[0] >= WINDOW_WIDTH or
                position[1] < 0 or position[1] >= WINDOW_HEIGHT or
                tuple(position) in map(tuple, snake_body))

    def update_direction(self, current_direction, action_index):
        actions = ["LEFT", "STRAIGHT", "RIGHT"]
        action = actions[action_index]
        if action == "LEFT":
            return LEFT_TURN[current_direction]
        elif action == "RIGHT":
            return RIGHT_TURN[current_direction]
        return current_direction

    def move_snake(self, snake_pos, snake_body, fruit_pos, fruit_spawn, direction, score):
        snake_pos[0] += DIRECTIONS[direction][0]
        snake_pos[1] += DIRECTIONS[direction][1]
        
        reward = -0.5
        done = False
        
        if snake_pos == fruit_pos:
            score += 1
            reward = 10
            fruit_spawn = False
        else:
            snake_body.pop()
        
        if not fruit_spawn:
            fruit_pos, fruit_spawn = self.generate_fruit(snake_body)
        
        if self.check_collision(snake_pos, snake_body):
            reward = -10
            done = True
        
        snake_body.insert(0, list(snake_pos))
        return snake_pos, fruit_pos, fruit_spawn, score, done, reward

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.dqn(state_tensor)
            return torch.argmax(q_values).item()

    def train_dqn(self):
        if len(REPLAY_MEMORY) < BATCH_SIZE:
            return
        
        batch = random.sample(REPLAY_MEMORY, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_tensor = torch.FloatTensor(np.array(states, dtype=np.float32))
        next_states_tensor = torch.FloatTensor(np.array(next_states, dtype=np.float32))
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)
        
        current_q_values = self.dqn(states_tensor).gather(1, actions_tensor).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.dqn(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (DISCOUNT_FACTOR * next_q_values * (1 - dones_tensor))
        
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_replay_memory(self, state, action, reward, next_state, done):
        REPLAY_MEMORY.append((state, action, reward, next_state, done))
        if len(REPLAY_MEMORY) > BATCH_SIZE:
            self.train_dqn()

    def train(self, num_games):
        for game_index in range(num_games):
            self.game_history.append(game_index)
            score = self.play_game(training=True)
            self.score_history.append(score)
            self.epsilon = max(self.epsilon * EPSILON_DECAY, 0.01)

            if game_index % 100 == 0:
                print(f"Game {game_index}, Score: {score}")

    def play_game(self, training=True):
        snake_pos, snake_body, direction, score = self.initialize_game()
        fruit_pos, fruit_spawn = self.generate_fruit(snake_body)
        
        while True:
            state = self.get_state(snake_pos, fruit_pos, snake_body, direction)
            action_index = self.choose_action(state)
            direction = self.update_direction(direction, action_index)
            
            snake_pos, fruit_pos, fruit_spawn, score, done, reward = self.move_snake(
                snake_pos, snake_body, fruit_pos, fruit_spawn, direction, score
            )
            
            next_state = self.get_state(snake_pos, fruit_pos, snake_body, direction)
            
            if training:
                self.update_replay_memory(state, action_index, reward, next_state, done)
            
            if done:
                break
                
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
        model_state = torch.load(filepath + "_model.pth", weights_only=True)
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
# Example usage
if __name__ == "__main__":
    game = SnakeGame()
    
    # Train the model
    start_time = time.time()
    game.train(1)
    print(f"Training completed in {(time.time() - start_time):.2f} seconds")
    
    # Save the model and results
    game.save_model("snake_model.pth")
    game.save_results_to_csv()