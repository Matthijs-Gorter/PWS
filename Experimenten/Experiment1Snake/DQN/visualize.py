import pygame
import numpy as np
import torch
import time
from train import DQNAgent, SnakeEnv, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, GRID_SIZE  # Import from train.py

# Pygame parameters
CELL_SIZE = 30
GRID_SIZE = 10
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE 

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Load the trained model
agent = DQNAgent(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
agent.policy_net.load_state_dict(torch.load("dqn_snake.pth"))
agent.policy_net.eval()

# Game loop
env = SnakeEnv(GRID_SIZE)
state = env.reset()
done = False

while not done:
    screen.fill(BLACK)

    # Get action from model
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = agent.policy_net(state_tensor).argmax().item()

    # Step in environment
    state, _, done, _ = env.step(action)

    # Draw snake
    for segment in env.snake:
        pygame.draw.rect(screen, GREEN, (segment[1] * CELL_SIZE, segment[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw food
    pygame.draw.rect(screen, RED, (env.food[1] * CELL_SIZE, env.food[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(5)  # Control game speed

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

pygame.quit()
