import pygame
import torch
import numpy as np
from train import SnakeGame, DQNetwork  # Assuming previous code is saved as snake_dqn.py

class SnakeGameVisualizer(SnakeGame):
    def __init__(self):
        super().__init__()
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption('Snake AI Visualization')
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Display settings
        self.game_window = pygame.display.set_mode((800, 600))
        self.fps_controller = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def render_game(self, snake_body, fruit_pos, score, current_state=None):
        """Render the game state"""
        self.game_window.fill(self.BLACK)
        
        # Draw snake
        for pos in snake_body:
            pygame.draw.rect(self.game_window, self.GREEN, 
                           pygame.Rect(pos[0], pos[1], 20, 20))
        
        # Draw fruit
        pygame.draw.rect(self.game_window, self.RED, 
                        pygame.Rect(fruit_pos[0], fruit_pos[1], 20, 20))
        
        # Draw score
        score_text = self.font.render(f'Score: {score}', True, self.WHITE)
        self.game_window.blit(score_text, (10, 10))
        
        # Draw state information if available
        if current_state is not None:
            state_text = self.font.render(
                f'State: {[round(x, 2) for x in current_state]}', 
                True, self.WHITE
            )
            self.game_window.blit(state_text, (10, 550))
        
        pygame.display.update()

    def play_game_visual(self, fps=10):
        """Play the game with visualization"""
        snake_pos, snake_body, direction, score = self.initialize_game()
        fruit_pos, fruit_spawn = self.generate_fruit(snake_body)
        
        running = True
        paused = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            if not paused:
                # Get current state and choose action
                state = self.get_state(snake_pos, fruit_pos, snake_body, direction)
                action_index = self.choose_action(state)
                direction = self.update_direction(direction, action_index)
                
                # Update game state
                snake_pos, fruit_pos, fruit_spawn, score, done, _ = self.move_snake(
                    snake_pos, snake_body, fruit_pos, fruit_spawn, direction, score
                )
                
                # Render the game
                self.render_game(snake_body, fruit_pos, score, state)
                
                if done:
                    break
                
                self.fps_controller.tick(fps)
            
            else:
                # Draw paused text
                pause_text = self.font.render('PAUSED', True, self.WHITE)
                pause_rect = pause_text.get_rect(center=(400, 300))
                self.game_window.blit(pause_text, pause_rect)
                pygame.display.update()
        
        pygame.quit()
        return score

def main():
    # Create visualizer and load trained model
    visualizer = SnakeGameVisualizer()
    try:
        visualizer.load_model("snake_model.pth")
        print("Loaded trained model successfully!")
    except FileNotFoundError:
        print("No trained model found. Using untrained model.")
    
    # Set a low epsilon for visualization (more exploitation)
    visualizer.epsilon = 0.01
    
    # Play multiple games
    num_games = 5
    scores = []
    
    for i in range(num_games):
        print(f"\nStarting Game {i+1}")
        score = visualizer.play_game_visual(fps=10)  # Adjust fps for speed
        scores.append(score)
        print(f"Game {i+1} Score: {score}")
    
    print("\nVisualization complete!")
    print(f"Average Score: {sum(scores)/len(scores):.2f}")
    print(f"Max Score: {max(scores)}")

if __name__ == "__main__":
    main()