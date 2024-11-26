import pygame
import time
import random
import numpy as np
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch_directml

#Device waarop DQN berekenen worden gedaan is gpu
device = torch_directml.device()

# Game
snakeSpeed = 15
windowX, windowY = 800, 600

# Logging
games, scores = [], []

# Colors
black, white, red, green = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)

# Pygame
pygame.init()
pygame.display.set_caption('Q_Learning_Snake')
gameWindow = pygame.display.set_mode((windowX, windowY))
FPS = pygame.time.Clock()

DIRECTIONS = {"UP": (0, -20), "DOWN": (0, 20), "LEFT": (-20, 0), "RIGHT": (20, 0)}
LEFT_TURN = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
RIGHT_TURN = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}


# Deep Q Learning
alpha, gamma, epsilon, epsilonDecay = 0.001, 0.95, 1, 0.995
nActions, nStates = 3, 6
replay_memory = deque(maxlen=10000)
Q = np.zeros((nStates, nActions))
batch_size = 64

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
dqn = DQNetwork(nStates, nActions).to(device)
optimizer = torch.optim.SGD(dqn.parameters(), lr=alpha)
criterion = nn.MSELoss()

def getState(snake_position, fruit_position, snake_body, direction):
    direction_vector = DIRECTIONS[direction]
    left_vector = DIRECTIONS[LEFT_TURN[direction]]
    right_vector = DIRECTIONS[RIGHT_TURN[direction]]

    food_direction = (fruit_position[0] - snake_position[0], fruit_position[1] - snake_position[1])

    state = [
        checkCollision([snake_position[0] + direction_vector[0], snake_position[1] + direction_vector[1]], snake_body),
        checkCollision([snake_position[0] + right_vector[0], snake_position[1] + right_vector[1]], snake_body),
        checkCollision([snake_position[0] + left_vector[0], snake_position[1] + left_vector[1]], snake_body),
        int(direction_vector[0] * food_direction[0] > 0 or direction_vector[1] * food_direction[1] > 0),
        int(right_vector[0] * food_direction[0] > 0 or right_vector[1] * food_direction[1] > 0),
        int(left_vector[0] * food_direction[0] > 0 or left_vector[1] * food_direction[1] > 0),
    ]
    return np.array(state, dtype=np.float32)



def checkCollision(position, snake_body):
    return (position[0] < 0 or position[0] >= windowX or
            position[1] < 0 or position[1] >= windowY or
            tuple(position) in snake_body)

def chooseAction(state):
    global epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(nActions)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = dqn(state_tensor)
            return torch.argmax(q_values).item()  
        
def updateReplayMemory(current_state, action, reward, next_state, done):
    replay_memory.append((current_state, action, reward, next_state, done))

def trainDQN():
    if len(replay_memory) < batch_size:
        return

    batch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Zet lijsten om naar numpy-arrays
    states_array = np.array(states, dtype=np.float32)
    next_states_array = np.array(next_states, dtype=np.float32)

    # Converteer naar tensors
    states_tensor = torch.FloatTensor(states_array).to(device)
    next_states_tensor = torch.FloatTensor(next_states_array).to(device)
    actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    dones_tensor = torch.FloatTensor(dones).to(device)


    # Current Q-values
    current_q_values = dqn(states_tensor).gather(1, actions_tensor).squeeze(1)

    # Target Q-values
    with torch.no_grad():
        next_q_values = dqn(next_states_tensor).max(1)[0]
        target_q_values = rewards_tensor + (gamma * next_q_values * (1 - dones_tensor))

    # Update the network
    loss = criterion(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def startNewGame(training):
    global epsilon
    snake_position = [300, 300]

    # Defining first 4 blocks of snake body
    snake_body = [[300, 300],
                [280, 300],
                [260, 300],
                [240, 300],
                [230, 300]]

    fruit_position = [random.randrange(1, (windowX // 20)) * 20,
                    random.randrange(1, (windowY // 20)) * 20]

    fruit_spawn = True
    direction = 'RIGHT'
    score = 0
    done = False
    turnsAfterLastApple = 0
    # Main Function
    while not done:
        current_state = getState(snake_position, fruit_position, snake_body, direction)
        actionIndex = chooseAction(current_state)
        action = ["LEFT", "STRAIGHT","RIGHT"][actionIndex]
        
        if action == 'LEFT':
            if direction == "UP":
                direction = "LEFT"
            elif direction == "DOWN":
                direction = "RIGHT"
            elif direction == "LEFT":
                direction = "DOWN"  # Snake turns down if it's already going left
            elif direction == "RIGHT":
                direction = "UP"

        elif action == 'RIGHT':
            if direction == "UP":
                direction = "RIGHT"
            elif direction == "DOWN":
                direction = "LEFT"
            elif direction == "LEFT":
                direction = "UP"
            elif direction == "RIGHT":
                direction = "DOWN"

        if direction == 'UP':
            snake_position[1] -= 20
        if direction == 'DOWN':
            snake_position[1] += 20
        if direction == 'LEFT':
            snake_position[0] -= 20
        if direction == 'RIGHT':
            snake_position[0] += 20

        reward = 0

        # Snake body growing 
        snake_body.insert(0, list(snake_position))
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 1
            reward = 10  
            fruit_spawn = False
            turnsAfterLastApple = 0
        else:
            snake_body.pop()

        if not fruit_spawn:
            while True:
                fruit_position = [random.randrange(1, (windowX // 20)) * 20,
                                random.randrange(1, (windowY // 20)) * 20]
                # Ensure fruit does not spawn in the snake's body
                if tuple(fruit_position) not in map(tuple, snake_body):
                    break

        fruit_spawn = True
        gameWindow.fill(black)

        if not training:    
            # Drawing the snake and the fruit
            for pos in snake_body:
                pygame.draw.rect(gameWindow, green, pygame.Rect(pos[0], pos[1], 20, 20))
            pygame.draw.rect(gameWindow, red, pygame.Rect(fruit_position[0], fruit_position[1], 20, 20))

        # Game Over conditions
        if snake_position[0] < 0 or snake_position[0] > windowX - 20 or snake_position[1] < 0 or snake_position[1] > windowY - 20 or turnsAfterLastApple > 300:
            reward = -10 
            gameOver(score,training)
            done = True

        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                reward = -10
                gameOver(score,training)
                done = True

        # Get next state after the action
        next_state = getState(snake_position, fruit_position, snake_body, direction)
    
        # Update the Q-table en the Replay memory 
        trainDQN()
        replay_memory.append((current_state, actionIndex, reward, next_state, done))

        # Frame Per Second / Refresh Rate
        if training:
            FPS.tick(10000000)
        else:
            # Displaying score continuously
            showScore(1, white, 'times new roman', 20, score)

            # Refresh game screen
            pygame.display.update()

            FPS.tick(snakeSpeed)
        
        if done:
            return score
        
        turnsAfterLastApple += 1

def showScore(choice, color, font, size, score):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    
    gameWindow.blit(score_surface, score_rect)

def gameOver(score, training):
    if not training:
        font = pygame.font.SysFont('times new roman', 50)
        gameOverText = font.render(
            'Your Score is : ' + str(score), True, red)
        gameOverRect = gameOverText.get_rect()
        gameOverRect.midtop = (windowX / 2, windowY / 4)
        gameWindow.blit(gameOverText, gameOverRect)
        pygame.display.flip()
        time.sleep(2)
        pygame.quit()
        quit()

def train(numGames):
    global epsilon
    for i in range(numGames):
        games.append(i)
        scores.append(startNewGame(True))
        epsilon *= epsilonDecay
        if i % 100 == 0:
            print(i)
            
start_time = time.time()            
train(300)    
print(f"Time elapsed: {(time.time() - start_time):.2f} seconds")
      
with open('games_scores.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Game', 'Score'])  # header
    writer.writerows(zip(games, scores))  # data

startNewGame(False)
