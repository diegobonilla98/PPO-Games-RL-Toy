import pygame
import random
import numpy as np
from PIL import Image
from scipy.spatial import distance


# Global configuration
SCREEN_SIZE = 600         # Square canvas (600x600 pixels)
GRID_SIZE   = 20          # Size of one grid cell (pixels)
CELL_COUNT  = SCREEN_SIZE // GRID_SIZE  # Number of cells per row/column
FPS         = 5          # Game update speed
DO_WARPING = False        # Whether to wrap around the screen

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        
        self.background = (0, 0, 0)
        
        self.difficulty = 5

        self.food_reward = 10
        self.crash_reward = -10
        self.alive_reward = 0.05  # 0.01
        self.heading_towards_food_reward = 0.01
        self.hunger_penalty = 0.0  # -0.001

        self.steps_without_fruit = 0
        
        self.last_reward = 0
        self.reset()

    def reset(self):
        """Resets the game state (snake position, direction, fruits)."""
        mid = CELL_COUNT // 2
        # Initialize a snake of length 3 with head at index 0
        self.snake = [(mid, mid), (mid, mid + 1), (mid, mid + 2)]
        self.direction = (0, -1)  # Moving up initially

        # Spawn multiple fruits based on difficulty
        self.spawn_fruits()

    def spawn_fruits(self):
        """Creates DIFFICULTY number of fruits at random positions."""
        self.fruits = []
        for _ in range(self.difficulty):
            self.spawn_single_fruit()

    def spawn_single_fruit(self):
        """Places a single fruit in a random location not occupied by snake or existing fruits."""
        while True:
            pos = (random.randint(0, CELL_COUNT - 1), random.randint(0, CELL_COUNT - 1))
            if pos not in self.snake and pos not in self.fruits:
                self.fruits.append(pos)
                break

    def step(self, action):
        """
        Takes an action (0=Up, 1=Right, 2=Down, 3=Left), updates the game state,
        and returns (reward, crashed).
        """
        self.steps_without_fruit += 1
        # Mapping of action to movement direction
        mapping = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0)
        }
        new_direction = mapping.get(action, self.direction)

        # Prevent immediate reversal if the snake length is > 1
        if len(self.snake) > 1:
            curr_dx, curr_dy = self.direction
            if (new_direction[0] == -curr_dx) and (new_direction[1] == -curr_dy):
                # Ignore illegal move (immediate 180 turn)
                new_direction = self.direction

        self.direction = new_direction

        # Move the snake
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Handle screen warping or boundary collision
        if DO_WARPING:
            new_head = (new_head[0] % CELL_COUNT, new_head[1] % CELL_COUNT)
        else:
            if not (0 <= new_head[0] < CELL_COUNT) or not (0 <= new_head[1] < CELL_COUNT):
                self.last_reward = self.crash_reward  # Penalize boundary collision
                self.reset()           # Reset the game
                return self.last_reward, True  # Collision occurred

        # Check for self-collision (except for last tail segment)
        if new_head in self.snake[:-1]:
            self.last_reward = self.crash_reward  # Penalize self-collision
            self.reset()           # Reset the game
            return self.last_reward, True  # Collision occurred

        # Insert new head
        self.snake.insert(0, new_head)

        # Check for fruit collision
        if new_head in self.fruits:
            # Reward for eating fruit
            self.fruits.remove(new_head)
            # Respawn only the eaten fruit
            self.spawn_single_fruit()
            self.last_reward = self.food_reward
            self.steps_without_fruit = 0
        else:
            # Normal movement - remove tail
            self.snake.pop()
            self.last_reward = self.alive_reward * (len(self.snake) - 3)  # Reward for staying alive (scaled by length)
        
        # Reward for being near food
        prev_head = np.array(self.snake[1])
        head = np.array(self.snake[0])
        distances_prev = [distance.cityblock(prev_head, (fx, fy)) for fx, fy in self.fruits]
        distances_now  = [distance.cityblock(head,      (fx, fy)) for fx, fy in self.fruits]
        min_prev_dist = min(distances_prev)
        min_new_dist  = min(distances_now)
        diff = min_prev_dist - min_new_dist  # >0 if we got closer
        if diff > 0:
            self.last_reward += self.heading_towards_food_reward * diff
        elif diff < 0:
            self.last_reward -= self.heading_towards_food_reward * abs(diff)
        
        # Penalize for not eating
        self.last_reward += self.hunger_penalty * self.steps_without_fruit

        return self.last_reward, False  # No collision

    def get_state_simple(self):
        """
        Returns a simplified state representation:
        - List of snake body segments
        - Head location
        - Snake length
        - List of fruit locations
        """
        state = {
            "snake_body": self.snake,
            "head_location": self.snake[0],
            "snake_length": len(self.snake),
            "fruits": self.fruits
        }
        return state

    def get_state(self):
        """
        Renders the current game state into a simple grid and returns it along with
        extra numerical features.
        
        'extra' includes:
          1) Snake length (scaled)
          2) Snake head coordinates (scaled)
          3) Distance to each apple (scaled)
        """
        # Initialize a 4-channel grid
        grid = np.zeros((4, CELL_COUNT, CELL_COUNT), dtype=np.float32)
        
        # Mark snake body (excluding head) in channel 1
        for x, y in self.snake[:-1]:
            grid[1, y, x] = 1.0
        
        # Mark snake head in channel 2
        if self.snake:
            head_x, head_y = self.snake[-1]
            grid[2, head_y, head_x] = 1.0
        
        # Mark fruits in channel 3
        for fx, fy in self.fruits:
            grid[3, fy, fx] = 1.0
        
        # Channel 0: Mark empty spaces where none of the other channels is active
        grid[0] = 1.0 - np.clip(np.sum(grid[1:], axis=0), 0, 1)


        # Snake head
        head = self.snake[0]
        head_x_scaled = head[0] / CELL_COUNT
        head_y_scaled = head[1] / CELL_COUNT

        # Scale all fruit coordinates and calculate distances
        distances_scaled = []
        for fx, fy in self.fruits:
            dist = distance.cityblock(head, (fx, fy))  # City block distance because of grid
            distances_scaled.append(dist / CELL_COUNT)

        # Construct extra info
        snake_len_scaled = len(self.snake) / 100.0
        extra = [
            snake_len_scaled,     # Snake length scaled
            head_x_scaled,        # Snake head x
            head_y_scaled,        # Snake head y
        ] + distances_scaled

        return grid, extra

    def draw(self):
        """Handles all the rendering: background, fruits, snake, and score."""
        self.screen.fill(self.background)

        # Draw each fruit as a bright red ellipse
        for fx, fy in self.fruits:
            fruit_rect = pygame.Rect(fx * GRID_SIZE, fy * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.ellipse(self.screen, (255, 0, 0), fruit_rect)
        
        # Draw each segment of the snake; make the head a brighter green
        for i, segment in enumerate(self.snake):
            seg_rect = pygame.Rect(segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(self.screen, color, seg_rect)

    def run(self):
        """Interactive game loop for manual play."""
        running = True
        k = 0
        while running:
            k += 1
            # Process events
            action = 4  # No move by default
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print(k)
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Map arrow keys to actions (0: up, 1: right, 2: down, 3: left)
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_LEFT:
                        action = 3
                    elif event.key == pygame.K_SPACE:
                        self.difficulty -= 1
                        print(self.difficulty)
                        self.reset()
            
            # Update game logic and render
            reward, crashed = self.step(action)
            print(f"Reward: {reward}")
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

# When run as a script, start interactive play.
if __name__ == "__main__":
    game = SnakeGame()
    game.run()
