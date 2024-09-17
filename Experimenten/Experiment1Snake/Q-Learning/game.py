import pygame
import time
import random
import numpy as np

snakeSpeed = 15

# Window size
windowX = 800
windowY = 600

# Q Learning
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
numActions = 3 
Q = {}  # Q table

# Defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# Initializing pygame
pygame.init()

# Initialize game window
pygame.display.set_caption('Q_Learning_Snake')
gameWindow = pygame.display.set_mode((windowX, windowY))

# FPS (frames per second) controller
FPS = pygame.time.Clock()

# action = int 1 = left; 2 = straight; 3 = right;

def getState(snake_position, fruit_position, snake_body, direction):
    pointLeft = [snake_position[0]  - 20, snake_position[1]]
    pointRight = [snake_position[0] + 20, snake_position[1]]
    pointUp = [snake_position[0]        , snake_position[1] + 20]
    pointDown = [snake_position[0]      , snake_position[1] - 20]
    
    foodLeft = fruit_position[0] < snake_position[0]
    foodRight = fruit_position[0] > snake_position[0]
    foodUp = fruit_position[1] < snake_position[1]
    foodDown = fruit_position[1] > snake_position[1] 
    
    return (
        # Danger straight
        (direction == "RIGHT" and checkCollision(pointRight,snake_body)) or 
        (direction == "LEFT" and checkCollision(pointLeft,snake_body)) or 
        (direction == "UP" and checkCollision(pointUp,snake_body)) or 
        (direction == "DOWN" and checkCollision(pointDown,snake_body)),

        # Danger right
        (direction == "UP" and checkCollision(pointRight,snake_body)) or 
        (direction == "DOWN" and checkCollision(pointLeft,snake_body)) or 
        (direction == "LEFT" and checkCollision(pointUp,snake_body)) or 
        (direction == "RIGHT" and checkCollision(pointDown,snake_body)),

        # Danger left
        (direction == "DOWN" and checkCollision(pointRight,snake_body)) or 
        (direction == "UP" and checkCollision(pointLeft,snake_body)) or 
        (direction == "RIGHT" and checkCollision(pointUp,snake_body)) or 
        (direction == "LEFT" and checkCollision(pointDown,snake_body)),
        
        # food straight
        (direction == "RIGHT" and foodRight) or 
        (direction == "LEFT" and foodLeft) or 
        (direction == "UP" and foodUp) or 
        (direction == "DOWN" and foodDown),

        # food right
        (direction == "UP" and foodRight) or 
        (direction == "DOWN" and foodLeft) or 
        (direction == "LEFT" and foodUp) or 
        (direction == "RIGHT" and foodDown),

        # food left
        (direction == "DOWN" and foodRight) or 
        (direction == "UP" and foodLeft) or 
        (direction == "RIGHT" and foodUp) or 
        (direction == "LEFT" and foodDown),
        
        # food behind
        (direction == "LEFT" and foodRight) or 
        (direction == "RIGHT" and foodLeft) or 
        (direction == "DOWN" and foodUp) or 
        (direction == "UP" and foodDown),
        
    )


def checkCollision(position,snake_body):
    if position[0] < 0 or position[0] >= windowX or position[1] < 0 or position[1] >= windowY:
        return True
    if position in snake_body:
        return True
    return False

def chooseAction(state): 
    if state not in Q:
        Q[state] = np.zeros(numActions)
    return np.argmax(Q[state])


def updateQTable(state, action, reward, next_state):
    if next_state not in Q:
        Q[next_state] = np.zeros(numActions)
    best_next_action = np.argmax(Q[next_state])
    Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])


def startNewGame():
    # Defining snake default position
    snake_position = [300, 300]

    # Defining first 4 blocks of snake body
    snake_body = [[300, 300],
                [280, 300],
                [260, 300],
                [240, 300],
                [230, 300]]

    # Fruit position
    fruit_position = [random.randrange(1, (windowX // 20)) * 20,
                    random.randrange(1, (windowY // 20)) * 20]

    fruit_spawn = True

    # Setting default snake direction towards right
    direction = 'RIGHT'
    change_to = direction

    # Initial score
    score = 0

    # Main Function
    while True:
        action = chooseAction(getState)


        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        # Moving the snake
        if direction == 'UP':
            snake_position[1] -= 20
        if direction == 'DOWN':
            snake_position[1] += 20
        if direction == 'LEFT':
            snake_position[0] -= 20
        if direction == 'RIGHT':
            snake_position[0] += 20

        # Snake body growing mechanism
        snake_body.insert(0, list(snake_position))
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 1
            fruit_spawn = False
        else:
            snake_body.pop()

        if not fruit_spawn:
            fruit_position = [random.randrange(1, (windowX // 20)) * 20,
                            random.randrange(1, (windowY // 20)) * 20]

        fruit_spawn = True
        gameWindow.fill(black)

        # Drawing the snake and the fruit
        for pos in snake_body:
            pygame.draw.rect(gameWindow, green, pygame.Rect(pos[0], pos[1], 20, 20))
        pygame.draw.rect(gameWindow, red, pygame.Rect(fruit_position[0], fruit_position[1], 20, 20))

        # Game Over conditions
        if snake_position[0] < 0 or snake_position[0] > windowX - 20:
            gameOver(score)
        if snake_position[1] < 0 or snake_position[1] > windowY - 20:
            gameOver(score)

        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                gameOver(score)

        # Displaying score continuously
        showScore(1, white, 'times new roman', 20, score)

        # Refresh game screen
        pygame.display.update()

        # Frame Per Second / Refresh Rate
        FPS.tick(snakeSpeed)

# Displaying score function
def showScore(choice, color, font, size, score):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    gameWindow.blit(score_surface, score_rect)

# Game over function
def gameOver(score):
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

# action = int 1 = left; 2 = straight; 3 = right;
# def getState(snake_position, fruit_position, snake_body, direction):
    
state = getState([300,300],[320,280],[[300,300]],"DOWN")
output = dict(zip(["Danger straight", "Danger right", "Danger left", "food straight", "food right", "food left", "food behind"], state))
print(output)