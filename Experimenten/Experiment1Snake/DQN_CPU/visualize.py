import pygame
import torch
import numpy as np
from snake_env import SnakeEnv
from model import DQNAgent
import time

def visualize_agent(model_path, pixel_size=10):
    # Initialize pygame
    pygame.init()
    
    # Initialize environment and agent
    env = SnakeEnv()
    state_shape = env.get_state_shape()
    n_actions = env.get_action_space()
    agent = DQNAgent(state_shape, n_actions)
    
    # Load trained model
    agent.load_model(model_path)
    agent.epsilon = 0  # No exploration during visualization
    
    # Set up display
    width = env.width * pixel_size
    height = env.height * pixel_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Snake DQN Visualization')
    
    # Color definitions
    BACKGROUND = (0, 0, 0)
    SNAKE_BODY = (0, 255, 0)
    SNAKE_HEAD = (0, 200, 0)
    FOOD = (255, 0, 0)
    
    running = True
    while running:
        state = env.reset()
        done = False
        score = 0
        
        while not done and running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            # Get action from agent
            state_tensor = agent.preprocess_state(state)
            action = agent.select_action(state_tensor)
            
            # Step environment
            next_state, reward, done = env.step(action.item())
            score += reward
            
            # Clear screen
            screen.fill(BACKGROUND)
            
            # Draw current state
            for i in range(env.height):
                for j in range(env.width):
                    # Draw snake body
                    if (j, i) in env.snake_body[1:]:
                        pygame.draw.rect(screen, SNAKE_BODY, 
                                       (j * pixel_size, i * pixel_size, 
                                        pixel_size, pixel_size))
                    # Draw snake head
                    elif (j, i) == env.snake_body[0]:
                        pygame.draw.rect(screen, SNAKE_HEAD,
                                       (j * pixel_size, i * pixel_size,
                                        pixel_size, pixel_size))
                    # Draw food
                    elif (j, i) == env.food_position:
                        pygame.draw.rect(screen, FOOD,
                                       (j * pixel_size, i * pixel_size,
                                        pixel_size, pixel_size))
            
            # Update display
            pygame.display.flip()
            
            # Control visualization speed
            time.sleep(0.1)
            
            # Update state
            state = next_state
            
            # Display score
            pygame.display.set_caption(f'Snake DQN Visualization - Score: {score}')
        
        # Wait a bit before starting new game
        time.sleep(1)
    
    pygame.quit()

if __name__ == "__main__":
    model_path = "best_snake_model.pth"  # Replace with your model path
    visualize_agent(model_path)