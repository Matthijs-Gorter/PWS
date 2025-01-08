import numpy as np
from collections import deque
from enum import Enum

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class SnakeEnv:
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        # Initialize empty grid
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Initialize snake in the middle of the grid
        self.head = np.array([self.height//2, self.width//2])
        self.snake = deque([self.head.copy()])
        self.direction = Direction.RIGHT
        
        # Place initial food
        self.place_food()
        
        # Update grid
        self.update_grid()
        
        return self.grid.copy()
    
    def place_food(self):
        while True:
            food = np.random.randint(0, [self.height, self.width])
            if not any(np.array_equal(food, s) for s in self.snake):
                self.food = food
                break
    
    def update_grid(self):
        self.grid.fill(0)
        # Draw snake body
        for segment in self.snake:
            self.grid[segment[0], segment[1]] = 128
        # Draw snake head
        self.grid[self.head[0], self.head[1]] = 255
        # Draw food
        self.grid[self.food[0], self.food[1]] = 64
    
    def step(self, action):
        # Convert action (0,1,2,3) to new direction
        if action == 0:  # Right
            if self.direction != Direction.LEFT:
                self.direction = Direction.RIGHT
        elif action == 1:  # Down
            if self.direction != Direction.UP:
                self.direction = Direction.DOWN
        elif action == 2:  # Left
            if self.direction != Direction.RIGHT:
                self.direction = Direction.LEFT
        elif action == 3:  # Up
            if self.direction != Direction.DOWN:
                self.direction = Direction.UP
        
        # Move snake
        old_head = self.head.copy()
        if self.direction == Direction.RIGHT:
            self.head[1] += 1
        elif self.direction == Direction.LEFT:
            self.head[1] -= 1
        elif self.direction == Direction.DOWN:
            self.head[0] += 1
        elif self.direction == Direction.UP:
            self.head[0] -= 1
        
        # Check if game is over
        # Hit wall
        if (self.head[0] < 0 or self.head[0] >= self.height or 
            self.head[1] < 0 or self.head[1] >= self.width):
            return self.grid.copy(), -1, True
        
        # Hit self
        if any(np.array_equal(self.head, s) for s in self.snake):
            return self.grid.copy(), -1, True
        
        # Update snake body
        self.snake.appendleft(self.head.copy())
        
        # Check if food is eaten
        reward = 0
        if np.array_equal(self.head, self.food):
            reward = 1
            self.place_food()
        else:
            self.snake.pop()
        
        # Update grid
        self.update_grid()
        
        return self.grid.copy(), reward, False

    def get_state_shape(self):
        return (1, self.height, self.width)  # Channel, Height, Width

    def get_action_space(self):
        return 4  # Right, Down, Left, Up