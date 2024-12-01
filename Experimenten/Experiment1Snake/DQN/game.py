import pygame
import time
import random
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Game settings
snakeSpeed = 15
windowWidth, windowHeight = 800, 600

# Logging
gameHistory, scoreHistory = [], []

# Colors
BLACK, WHITE, RED, GREEN = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)

# Pygame initialization
pygame.init()
pygame.display.set_caption('Q_Learning_Snake')
gameWindow = pygame.display.set_mode((windowWidth, windowHeight))
fpsClock = pygame.time.Clock()

# Directions
DIRECTIONS = {"UP": (0, -20), "DOWN": (0, 20), "LEFT": (-20, 0), "RIGHT": (20, 0)}
LEFT_TURN = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
RIGHT_TURN = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}

# Deep Q-Learning settings
learningRate, discountFactor, epsilon, epsilonDecay = 0.005, 0.97, 1, 0.995
numActions, numStates = 3, 6
replayMemory = deque(maxlen=50000)
batchSize = 256

# Neural network for DQN
class DQNetwork(nn.Module):
    def __init__(self, stateSize, actionSize):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(stateSize, 128)
        self.fc2 = nn.Linear(128, actionSize)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

dqn = DQNetwork(numStates, numActions)
optimizer = optim.Adam(dqn.parameters(), lr=learningRate)
criterion = nn.MSELoss()

# debug
p1Time = 0
p2Time = 0
p3Time = 0
p4Time = 0
gameTime = 0 

def initializeGame():
    """Initialize the snake, body, direction, and score."""
    snakePos = [300, 300]
    snakeBody = [[300, 300], [280, 300], [260, 300], [240, 300]]
    direction = "RIGHT"
    score = 0
    return snakePos, snakeBody, direction, score

def getState(snakePos, fruitPos, snakeBody, direction):
    """Get the current state representation for the agent."""
    directionVector = DIRECTIONS[direction]
    leftVector = DIRECTIONS[LEFT_TURN[direction]]
    rightVector = DIRECTIONS[RIGHT_TURN[direction]]

    foodDirection = (fruitPos[0] - snakePos[0], fruitPos[1] - snakePos[1])

    state = [
        checkCollision([snakePos[0] + directionVector[0], snakePos[1] + directionVector[1]], snakeBody),
        checkCollision([snakePos[0] + rightVector[0], snakePos[1] + rightVector[1]], snakeBody),
        checkCollision([snakePos[0] + leftVector[0], snakePos[1] + leftVector[1]], snakeBody),
        int(np.sign(directionVector[0]) == np.sign(foodDirection[0]) or 
            np.sign(directionVector[1]) == np.sign(foodDirection[1])),
        int(np.sign(rightVector[0]) == np.sign(foodDirection[0]) or 
            np.sign(rightVector[1]) == np.sign(foodDirection[1])),
        int(np.sign(leftVector[0]) == np.sign(foodDirection[0]) or 
            np.sign(leftVector[1]) == np.sign(foodDirection[1])),
    ]
    return np.array(state, dtype=np.float32)

def generateFruit(snakeBody):
    """Generate a fruit position that does not overlap the snake."""
    while True:
        fruitPos = [random.randrange(1, (windowWidth // 20)) * 20, random.randrange(1, (windowHeight // 20)) * 20]
        if tuple(fruitPos) not in map(tuple, snakeBody):
            break
    return fruitPos, True

def updateDirection(currentDirection, actionIndex):
    """Update the snake's direction based on the action index."""
    actions = ["LEFT", "STRAIGHT", "RIGHT"]
    action = actions[actionIndex]
    if action == "LEFT":
        return LEFT_TURN[currentDirection]
    elif action == "RIGHT":
        return RIGHT_TURN[currentDirection]
    return currentDirection

def checkCollision(position, snakeBody):
    """Check if a position collides with the wall or the snake's body."""
    return (position[0] < 0 or position[0] >= windowWidth or
            position[1] < 0 or position[1] >= windowHeight or
            tuple(position) in snakeBody)

def moveSnake(snakePos, snakeBody, fruitPos, fruitSpawn, direction, score):
    """Move the snake and update the game state."""
    snakePos[0] += DIRECTIONS[direction][0]
    snakePos[1] += DIRECTIONS[direction][1]

    reward = -0.5  # Default reward for surviving
    done = False

    # Check if snake eats the fruit
    if snakePos == fruitPos:
        score += 1
        reward = 10
        fruitSpawn = False
    else:
        snakeBody.pop()

    # Spawn new fruit if eaten
    if not fruitSpawn:
        fruitPos, fruitSpawn = generateFruit(snakeBody)

    # Check for collisions
    if checkCollision(snakePos, snakeBody):
        reward = -10
        done = True

    # Add the new head position
    snakeBody.insert(0, list(snakePos))

    return snakePos, fruitPos, fruitSpawn, score, done, reward

def updateReplayMemory(state, action, reward, nextState, done):
    """Store experiences in replay memory and train the network."""
    replayMemory.append((state, action, reward, nextState, done))
    if len(replayMemory) > batchSize:
        trainDQN()
        
def chooseAction(state):
    """Choose an action using epsilon-greedy policy."""
    global epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(numActions)
    else:
        with torch.no_grad():
            stateTensor = torch.FloatTensor(state).unsqueeze(0)
            qValues = dqn(stateTensor)
            return torch.argmax(qValues).item()
        
def trainDQN():
    """Train the DQN model using replay memory."""
    if len(replayMemory) < batchSize:
        return
    
    batch = random.sample(replayMemory, batchSize)
    states, actions, rewards, nextStates, dones = zip(*batch)

    # Convert to tensors
    statesTensor = torch.FloatTensor(np.array(states, dtype=np.float32))
    nextStatesTensor = torch.FloatTensor(np.array(nextStates, dtype=np.float32))
    actionsTensor = torch.LongTensor(actions).unsqueeze(1)
    rewardsTensor = torch.FloatTensor(rewards)
    donesTensor = torch.FloatTensor(dones)

    # Compute current Q-values
    currentQValues = dqn(statesTensor).gather(1, actionsTensor).squeeze(1)

    # Compute target Q-values
    with torch.no_grad():
        nextQValues = dqn(nextStatesTensor).max(1)[0]
        targetQValues = rewardsTensor + (discountFactor * nextQValues * (1 - donesTensor))

    # Optimize the network
    loss = criterion(currentQValues, targetQValues)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def renderGame(snakeBody, fruitPos, score):
    """Render the game using pygame."""
    gameWindow.fill(BLACK)

    # Draw the snake
    for pos in snakeBody:
        pygame.draw.rect(gameWindow, GREEN, pygame.Rect(pos[0], pos[1], 20, 20))

    # Draw the fruit
    pygame.draw.rect(gameWindow, RED, pygame.Rect(fruitPos[0], fruitPos[1], 20, 20))

    # Display the score
    showScore(1, WHITE, 'times new roman', 20, score)

    # Refresh the display
    pygame.display.update()

def showScore(choice, color, font, size, score):
    """Display the score on the game window."""
    scoreFont = pygame.font.SysFont(font, size)
    scoreSurface = scoreFont.render(f'Score: {score}', True, color)
    scoreRect = scoreSurface.get_rect()
    gameWindow.blit(scoreSurface, scoreRect)

def train(numGames):
    """Train the DQN model by playing multiple games."""
    global epsilon
    pygame.time.Clock().tick(10000000)
    for gameIndex in range(numGames):
        gameHistory.append(gameIndex)
        score = startGame(isTraining=True)
        scoreHistory.append(score)
        epsilon = max(epsilon * epsilonDecay, 0.01)  # Ensure epsilon does not reach 0

        if gameIndex % 100 == 0:
            print(f"Game {gameIndex}, Score: {score}")

def saveResultsToCSV(filename="games_scores.csv"):
    """Save game results to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Game', 'Score'])
        writer.writerows(zip(gameHistory, scoreHistory))

def startGame(isTraining):
    """Start a new game and manage game loop."""
    global epsilon
    global p1Time
    global p2Time
    global p3Time
    global p4Time
    global gameTime
    snakePos, snakeBody, direction, score = initializeGame()
    fruitPos, fruitSpawn = generateFruit(snakeBody)

    startGameTime = time.time()
    while True:
        startTime = time.time()
        state = getState(snakePos, fruitPos, snakeBody, direction)
        actionIndex = chooseAction(state)
        direction = updateDirection(direction, actionIndex)
        p1Time += time.time() - startTime

        startTime = time.time()
        snakePos, fruitPos, fruitSpawn, score, done, reward = moveSnake(
            snakePos, snakeBody, fruitPos, fruitSpawn, direction, score
        )
        p2Time += time.time() - startTime

        startTime = time.time()
        nextState = getState(snakePos, fruitPos, snakeBody, direction)
        p3Time += time.time() - startTime
        
        startTime = time.time()
        updateR eplayMemory(state, actionIndex, reward, nextState, done)
        p4Time += time.time() - startTime
        if not isTraining:
            renderGame(snakeBody, fruitPos, score)
        
        if done:
            break
    gameTime += time.time() - startGameTime
    return score

# Start training
startTime = time.time()
train(100)
print(f"Training completed in {(time.time() - startTime):.2f} seconds")
print("game",gameTime)
print(1,p1Time)
print(2,p2Time)
print(3,p3Time)
print(4,p4Time)
# Save results and start a human-playable game
saveResultsToCSV()
startGame(isTraining=False)
