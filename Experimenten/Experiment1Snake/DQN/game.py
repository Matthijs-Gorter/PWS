import pygame
import torch
import numpy as np
from train import SnakeGame, DQNetwork, WINDOW_WIDTH, WINDOW_HEIGHT, GRID_SIZE, DIRECTIONS  # Import constants

class SnakeGameVisualizer(SnakeGame):
    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.display.set_caption('Snake AI Visualization')
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Scale factor for visualization (makes the game window larger than the pixel state)
        self.SCALE = 10
        
        # Display settings
        self.DISPLAY_WIDTH = WINDOW_WIDTH * self.SCALE
        self.DISPLAY_HEIGHT = WINDOW_HEIGHT * self.SCALE
        self.game_window = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT + 50))  # Extra space for score
        self.fps_controller = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def render_game(self, snake_body, fruit_pos, score, current_state=None):
        """Render the game state"""
        self.game_window.fill(self.BLACK)
        
        # Draw snake
        for pos in snake_body:
            pygame.draw.rect(self.game_window, self.GREEN, 
                           pygame.Rect(pos[0] * self.SCALE // GRID_SIZE, 
                                     pos[1] * self.SCALE // GRID_SIZE, 
                                     self.SCALE, self.SCALE))
        
        # Draw fruit
        pygame.draw.rect(self.game_window, self.RED, 
                        pygame.Rect(fruit_pos[0] * self.SCALE // GRID_SIZE, 
                                  fruit_pos[1] * self.SCALE // GRID_SIZE, 
                                  self.SCALE, self.SCALE))
        
        # Draw score
        score_text = self.font.render(f'Score: {score}', True, self.WHITE)
        self.game_window.blit(score_text, (10, self.DISPLAY_HEIGHT + 10))
        
        # Visualize the pixel state if available
        if current_state is not None:
            # Draw a small version of the pixel state in the corner
            pixel_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            state_array = current_state[0]  # Get the first channel
            
            # Convert state values to RGB
            for y in range(WINDOW_HEIGHT):
                for x in range(WINDOW_WIDTH):
                    value = int(state_array[y, x] * 255)
                    pixel_surface.set_at((x, y), (value, value, value))
            
            # Scale up the pixel state visualization
            pixel_surface = pygame.transform.scale(pixel_surface, 
                                                 (WINDOW_WIDTH * 2, WINDOW_HEIGHT * 2))
            self.game_window.blit(pixel_surface, (self.DISPLAY_WIDTH - WINDOW_WIDTH * 2, 0))
        
        pygame.display.update()

    def play_game_visual(self, fps=10):
        """Play the game with visualization"""
        snake_pos, snake_body, direction, score = self.initialize_game()
        fruit_pos, fruit_spawn = self.generate_fruit(snake_body)
        
        running = True
        paused = False
        steps_without_fruit = 0
        
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
                # Get current state
                current_state = self.get_state(snake_body, fruit_pos, direction)
                
                # Choose action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                    action_index = self.dqn(state_tensor).argmax().item()
                
                # Update direction
                direction = self.update_direction(direction, action_index)
                new_pos = [
                    snake_pos[0] + DIRECTIONS[direction][0],
                    snake_pos[1] + DIRECTIONS[direction][1]
                ]
                
                # Check for collision
                if self.check_collision(new_pos, snake_body):
                    break
                
                # Update snake position
                snake_pos = new_pos
                snake_body.insert(0, list(snake_pos))
                
                # Check if fruit eaten
                if snake_pos == fruit_pos:
                    score += 1
                    steps_without_fruit = 0
                    fruit_spawn = False
                else:
                    snake_body.pop()
                    steps_without_fruit += 1
                
                # Generate new fruit if needed
                if not fruit_spawn:
                    fruit_pos, fruit_spawn = self.generate_fruit(snake_body)
                
                # Render the game
                self.render_game(snake_body, fruit_pos, score, current_state)
                
                if steps_without_fruit > 300:
                    break
                
                self.fps_controller.tick(fps)
            
            else:
                # Draw paused text
                pause_text = self.font.render('PAUSED', True, self.WHITE)
                pause_rect = pause_text.get_rect(center=(self.DISPLAY_WIDTH//2, self.DISPLAY_HEIGHT//2))
                self.game_window.blit(pause_text, pause_rect)
                pygame.display.update()
        
        pygame.quit()
        return score

def main():
    # Create visualizer and load trained model
    visualizer = SnakeGameVisualizer()
    try:
        visualizer.load_model("snake_model_pixel")
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
        score = visualizer.play_game_visual(fps=10)
        scores.append(score)
        print(f"Game {i+1} Score: {score}")
    
    print("\nVisualization complete!")
    print(f"Average Score: {sum(scores)/len(scores):.2f}")
    print(f"Max Score: {max(scores)}")

if __name__ == "__main__":
    main()