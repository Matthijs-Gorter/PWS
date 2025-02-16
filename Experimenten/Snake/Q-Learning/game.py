import pygame
import time
import random
import numpy as np
import csv

# Grid instellingen voor een 10x10 grid
grid_size = 10      # 10 x 10 grid
cell_size = 20      # Elke cel is 20x20 pixels
windowX, windowY = grid_size * cell_size, grid_size * cell_size  # 200x200

# Q-Learning parameters
alpha, gamma, epsilon, epsilonDecay = 0.001, 0.95, 1, 0.995
nActions, nStates = 3, 128
Q = np.zeros((nStates, nActions))

# Global variabelen voor TD-loss logging (per episode)
cumulative_loss = 0
update_count = 0

# Logging voor episodes
log_data = []  # (Episode, TotalReward, ApplesEaten, AvgLoss, Epsilon, StepsPerApple, EpisodeTime)

# Kleuren
black, white, red, green = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)

# Pygame initialisatie
pygame.init()
pygame.display.set_caption('Q_Learning_Snake 10x10 Grid')
gameWindow = pygame.display.set_mode((windowX, windowY))
FPS = pygame.time.Clock()

DIRECTIONS = {"UP": (0, -cell_size), "DOWN": (0, cell_size), "LEFT": (-cell_size, 0), "RIGHT": (cell_size, 0)}
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
    # Epsilon-greedy actie-selectie
    if np.random.rand() < epsilon or np.all(Q[state] == 0):
        return np.random.randint(nActions)
    else:
        return np.argmax(Q[state])

def updateQTable(state, action, reward, next_state):
    global cumulative_loss, update_count
    td_error = reward + gamma * np.max(Q[next_state]) - Q[state][action]
    cumulative_loss += abs(td_error)
    update_count += 1
    Q[state][action] = Q[state][action] + alpha * td_error

def startNewGame(training):
    global cumulative_loss, update_count, epsilon

    # Startpositie en snake body voor een 10x10 grid
    start_x = (grid_size // 2) * cell_size  # Bijvoorbeeld 5 * 20 = 100
    start_y = (grid_size // 2) * cell_size
    snake_position = [start_x, start_y]
    snake_body = [
        [start_x, start_y],
        [start_x - cell_size, start_y],
        [start_x - 2 * cell_size, start_y]
    ]
    
    # Fruitpositie (zorg dat het binnen het grid ligt en niet op de snake)
    fruit_position = [random.randrange(0, grid_size) * cell_size,
                      random.randrange(0, grid_size) * cell_size]
    while tuple(fruit_position) in map(tuple, snake_body):
        fruit_position = [random.randrange(0, grid_size) * cell_size,
                          random.randrange(0, grid_size) * cell_size]

    fruit_spawn = True
    direction = 'RIGHT'
    score = 0           # Aantal appels
    total_reward = 0    # Totale reward in deze episode
    gameIsOver = False
    turnsAfterLastApple = 0
    steps_counter = 0

    # Reset TD-loss variabelen per episode
    cumulative_loss = 0
    update_count = 0

    while True:
        steps_counter += 1
        current_state = getState(snake_position, fruit_position, snake_body, direction)
        actionIndex = chooseAction(current_state)
        action = ["LEFT", "STRAIGHT", "RIGHT"][actionIndex]

        # Pas de richting aan op basis van de actie
        if action == 'LEFT':
            if direction == "UP":
                direction = "LEFT"
            elif direction == "DOWN":
                direction = "RIGHT"
            elif direction == "LEFT":
                direction = "DOWN"
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
        # Bij 'STRAIGHT' blijft de richting onveranderd

        # Beweeg de snake
        if direction == 'UP':
            snake_position[1] -= cell_size
        if direction == 'DOWN':
            snake_position[1] += cell_size
        if direction == 'LEFT':
            snake_position[0] -= cell_size
        if direction == 'RIGHT':
            snake_position[0] += cell_size

        reward = 0

        # Update snake body
        snake_body.insert(0, list(snake_position))
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 1
            reward = 10
            total_reward += 10
            fruit_spawn = False
            turnsAfterLastApple = 0
        else:
            snake_body.pop()
            total_reward += reward  # Meestal 0 (tenzij negatieve reward)

        # Nieuwe fruitpositie als nodig
        if not fruit_spawn:
            fruit_position = [random.randrange(0, grid_size) * cell_size,
                              random.randrange(0, grid_size) * cell_size]
            while tuple(fruit_position) in map(tuple, snake_body):
                fruit_position = [random.randrange(0, grid_size) * cell_size,
                                  random.randrange(0, grid_size) * cell_size]
        fruit_spawn = True

        gameWindow.fill(black)
        if not training:
            # Teken snake en fruit
            for pos in snake_body:
                pygame.draw.rect(gameWindow, green, pygame.Rect(pos[0], pos[1], cell_size, cell_size))
            pygame.draw.rect(gameWindow, red, pygame.Rect(fruit_position[0], fruit_position[1], cell_size, cell_size))

        # Controleer op game-over: botsing met muren of als er te lang geen appel wordt gegeten
        if (snake_position[0] < 0 or snake_position[0] >= windowX or 
            snake_position[1] < 0 or snake_position[1] >= windowY or 
            turnsAfterLastApple > 300):
            reward = -10
            total_reward += reward
            gameOver(score, training)
            gameIsOver = True

        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                reward = -10
                total_reward += reward
                gameOver(score, training)
                gameIsOver = True

        next_state = getState(snake_position, fruit_position, snake_body, direction)
        updateQTable(current_state, actionIndex, reward, next_state)

        if training:
            FPS.tick(10000000)
        else:
            showScore(1, white, 'times new roman', 20, score)
            pygame.display.update()
            FPS.tick(15)  # Lagere snelheid voor visuele weergave

        if gameIsOver:
            break

        turnsAfterLastApple += 1


    avg_loss = cumulative_loss / update_count if update_count > 0 else 0
    steps_per_apple = steps_counter / score if score > 0 else steps_counter
    return total_reward, score, avg_loss, epsilon, steps_per_apple

def showScore(choice, color, font, size, score):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    gameWindow.blit(score_surface, score_rect)

def gameOver(score, training):
    if not training:
        font = pygame.font.SysFont('times new roman', 50)
        gameOverText = font.render('Your Score is : ' + str(score), True, red)
        gameOverRect = gameOverText.get_rect()
        gameOverRect.midtop = (windowX / 2, windowY / 4)
        gameWindow.blit(gameOverText, gameOverRect)
        pygame.display.flip()
        time.sleep(2)
        pygame.quit()
        quit()

def train(numGames):
    global epsilon
    start_time = time.time()
    for i in range(numGames):
        total_reward, apples_eaten, avg_loss, current_epsilon, steps_per_apple = startNewGame(training=True)
        log_data.append((i, total_reward, apples_eaten, avg_loss, current_epsilon, steps_per_apple, time.time() - start_time))
        epsilon *= epsilonDecay/
        if i % 100 == 0:
            print(f"Episode {i}")

train(15000)

# Schrijf de loggegevens naar CSV
with open('games_scores.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Episode', 'TotalReward', 'ApplesEaten', 'AvgLoss', 'Epsilon', 'StepsPerApple', 'TotalTime'])
    writer.writerows(log_data)

# Start de visuele game na de training
startNewGame(training=False)
