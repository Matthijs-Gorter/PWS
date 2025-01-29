import pygame
import time
import random
import numpy as np
import time
import csv


# Game
snakeSpeed = 15
windowX, windowY = 800, 600

# Q Learning
alpha, gamma, epsilon, epsilonDecay = 0.001, 0.95, 1, 0.995
nActions, nStates = 3, 128
Q = np.zeros((nStates, nActions))

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


def getState(snake_position, fruit_position, snake_body, direction):
    direction_vector = DIRECTIONS[direction]
    left_vector = DIRECTIONS[LEFT_TURN[direction]]
    right_vector = DIRECTIONS[RIGHT_TURN[direction]]

    food_direction = (fruit_position[0] - snake_position[0], fruit_position[1] - snake_position[1])

    state = (
        # Danger straight
        checkCollision([snake_position[0] + direction_vector[0], snake_position[1] + direction_vector[1]], snake_body),

        # Danger right
        checkCollision([snake_position[0] + right_vector[0], snake_position[1] + right_vector[1]], snake_body),

        # Danger left
        checkCollision([snake_position[0] + left_vector[0], snake_position[1] + left_vector[1]], snake_body),

        # Food straight
        (direction_vector[0] * food_direction[0] > 0 or direction_vector[1] * food_direction[1] > 0),

        # Food right
        (right_vector[0] * food_direction[0] > 0 or right_vector[1] * food_direction[1] > 0),

        # Food left
        (left_vector[0] * food_direction[0] > 0 or left_vector[1] * food_direction[1] > 0),
    )
    return sum(1 << i for i, val in enumerate(state) if val)

def checkCollision(position, snake_body):
    return (position[0] < 0 or position[0] >= windowX or
            position[1] < 0 or position[1] >= windowY or
            tuple(position) in snake_body)

def chooseAction(state):
    # Epsilon-greedy action selection
    if np.random.rand() < epsilon or np.all(Q[state] == 0):
        # Exploration: choose a random action
        return np.random.randint(nActions)
    else:
        # Exploitation: choose the action with the highest Q-value
        return np.argmax(Q[state])

def updateQTable(state, action, reward, next_state):    
    # Q-learning update rule
    Q[state][action] = Q[state][action] + alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state][action]
    )

def startNewGame(training):
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
    gameIsOver = False
    turnsAfterLastApple = 0
    # Main Function
    while True:
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
            gameIsOver = True

        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                reward = -10
                gameOver(score,training)
                gameIsOver = True

        # Get next state after the action
        next_state = getState(snake_position, fruit_position, snake_body, direction)
    
        # Update the Q-table
        updateQTable(current_state, actionIndex, reward, next_state)
        
        # Frame Per Second / Refresh Rate
        if training:
            FPS.tick(10000000)
        else:
            # Displaying score continuously
            showScore(1, white, 'times new roman', 20, score)

            # Refresh game screen
            pygame.display.update()

            FPS.tick(snakeSpeed)
        
        if gameIsOver:
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
train(10000)    
print(f"Time elapsed: {(time.time() - start_time):.2f} seconds")
      
with open('games_scores.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Game', 'Score'])  # header
    writer.writerows(zip(games, scores))  # data

startNewGame(False)
