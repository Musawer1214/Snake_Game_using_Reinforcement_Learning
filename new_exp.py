import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----------------------------------------------------------------
# Global Parameters (Experiment Toggles)
# ----------------------------------------------------------------
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20

VISION_SIZE = 11    # local vision grid dimension (must be odd)
FRAME_STACK = 4     # how many consecutive frames to stack

NUM_RED_FOODS = 1   # how many red foods to place
NUM_BLUE_FOODS = 1  # how many blue foods to place
NUM_OBSTACLES = 5   # how many obstacles to place

show_vision_overlay = True  # toggle debug overlay for the local grid

device = torch.device("cpu")  # Force CPU usage

# ----------------------------------------------------------------
# Colors
# ----------------------------------------------------------------
COLOR_BG = (30, 30, 30)
COLOR_SNAKE_BODY = (0, 180, 0)
COLOR_SNAKE_HEAD = (0, 255, 0)
COLOR_RED_FOOD = (255, 50, 50)   # reward 10
COLOR_BLUE_FOOD = (50, 50, 255)  # reward 15
COLOR_OBSTACLE = (100, 100, 100)
COLOR_WALL = (120, 120, 120)     # used in debug overlay for out-of-bounds/walls
COLOR_EMPTY = (200, 200, 200)    # used in debug overlay for empty
COLOR_SCOREBAR = (60, 60, 60)
COLOR_TEXT = (255, 255, 255)
COLOR_GAMEOVER = (200, 50, 50)

def get_font(size=20, bold=False):
    """Helper to load a system font, fallback to default if not found."""
    return pygame.font.SysFont("arial", size, bold=bold)

# ----------------------------------------------------------------
# Snake Game Environment
# ----------------------------------------------------------------
class SnakeGameEnv:
    """
    Supports:
      - Multiple food items (red/blue), each re-spawns after being eaten.
      - Obstacles (collision => game over).
      - Local vision grid + frame stacking (CNN-based).
    """
    def __init__(self):
        self.direction = 0  # 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        self.snake = None
        self.head = None
        self.score = 0
        self.game_over = False

        # We'll store all foods in a list of (x, y, color, reward).
        # We'll also store obstacles as a set of (x, y).
        self.food_items = []
        self.obstacles = set()

        # For local vision grid
        self.grid_stack = deque(maxlen=FRAME_STACK)

        self.reset()

    def reset(self):
        """Reset the environment for a new episode."""
        self.direction = 0
        self.score = 0
        self.game_over = False

        # Center of screen
        mid_x = WIDTH // 2
        mid_y = HEIGHT // 2

        # Initial snake (3 blocks horizontal)
        self.snake = [
            (mid_x, mid_y),
            (mid_x - BLOCK_SIZE, mid_y),
            (mid_x - 2 * BLOCK_SIZE, mid_y),
        ]
        self.head = self.snake[0]

        # Clear food & obstacles
        self.food_items = []
        self.obstacles = set()

        # Place foods
        self._place_foods()

        # Place obstacles
        self._place_obstacles()

        # Clear frame stack
        self.grid_stack.clear()
        empty_grid = np.zeros((VISION_SIZE, VISION_SIZE), dtype=np.float32)
        for _ in range(FRAME_STACK):
            self.grid_stack.append(empty_grid)

        return self.get_state()

    def _place_foods(self):
        """Place multiple red & blue foods randomly on the board."""
        for _ in range(NUM_RED_FOODS):
            self._spawn_food(reward=10, color='red')

        for _ in range(NUM_BLUE_FOODS):
            self._spawn_food(reward=15, color='blue')

    def _spawn_food(self, reward, color):
        """Spawn a single food with given reward & color, not on snake/obstacle."""
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake and (x, y) not in self.obstacles:
                # food_items = [(x, y, color, reward), ...]
                self.food_items.append((x, y, color, reward))
                return

    def _place_obstacles(self):
        """Place a fixed number of obstacles on empty squares."""
        for _ in range(NUM_OBSTACLES):
            while True:
                x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                if (x, y) not in self.snake and (x, y) not in [f[:2] for f in self.food_items]:
                    self.obstacles.add((x, y))
                    break

    def step(self, action):
        """
        action: 0=straight, 1=turn right, 2=turn left
        returns: (next_state, reward, done)
        """
        # 1. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 2. Collision checks
        reward = 0

        # If out of bounds or snake-body collision => game over
        if self._collided_wall(self.head[0], self.head[1]) or (self.head in self.snake[1:]):
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True

        # If obstacle collision => game over
        if self.head in self.obstacles:
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True

        # 3. Check if food eaten
        eaten_idx = -1
        for i, (fx, fy, fcolor, freward) in enumerate(self.food_items):
            if (fx, fy) == self.head:
                eaten_idx = i
                self.score += 1
                reward = freward
                break

        if eaten_idx >= 0:
            # Instead of removing the food permanently, respawn it at a new location
            x_old, y_old, color_old, reward_old = self.food_items[eaten_idx]
            # Teleport that food to a new random spot
            self.food_items[eaten_idx] = self._random_new_food(color_old, reward_old)
        else:
            # move tail
            self.snake.pop()

        next_state = self.get_state()
        done = False
        return next_state, reward, done

    def _random_new_food(self, color, reward):
        """
        Teleport an existing food to a new random location.
        Return (x, y, color, reward).
        """
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            # must not be on snake or obstacle
            if (x, y) not in self.snake and (x, y) not in self.obstacles:
                return (x, y, color, reward)

    def _collided_wall(self, x, y):
        """Check if outside the board boundary."""
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return False

    def _move(self, action):
        """
        Update direction, then move head one block.
        directions: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        """
        dirs = [0, 1, 2, 3]
        idx = dirs.index(self.direction)

        if action == 1:   # turn right
            idx = (idx + 1) % 4
        elif action == 2: # turn left
            idx = (idx - 1) % 4

        self.direction = dirs[idx]

        x, y = self.head
        if self.direction == 0:   # RIGHT
            x += BLOCK_SIZE
        elif self.direction == 1: # DOWN
            y += BLOCK_SIZE
        elif self.direction == 2: # LEFT
            x -= BLOCK_SIZE
        elif self.direction == 3: # UP
            y -= BLOCK_SIZE

        self.head = (x, y)

    def get_state(self):
        """
        Return the stacked frames of the local vision grid (FRAME_STACK, VISION_SIZE, VISION_SIZE).
        """
        grid = self._get_vision_grid()
        self.grid_stack.append(grid)
        stacked_state = np.stack(self.grid_stack, axis=0)  # shape => (FRAME_STACK, VISION_SIZE, VISION_SIZE)
        return stacked_state

    def _get_vision_grid(self):
        """
        Create a (VISION_SIZE x VISION_SIZE) grid around the snake's head.
        Encoding:
          0 = empty
          1 = snake body
          2 = red food
          3 = wall or out-of-bounds
          4 = blue food
          5 = obstacle
        """
        grid = np.zeros((VISION_SIZE, VISION_SIZE), dtype=np.float32)
        radius = VISION_SIZE // 2

        hx, hy = self.head

        for cx in range(-radius, radius + 1):
            for cy in range(-radius, radius + 1):
                gx = cx + radius
                gy = cy + radius

                board_x = hx + cx * BLOCK_SIZE
                board_y = hy + cy * BLOCK_SIZE

                if board_x < 0 or board_x >= WIDTH or board_y < 0 or board_y >= HEIGHT:
                    # out of bounds => 3
                    grid[gy, gx] = 3
                else:
                    # check obstacle
                    if (board_x, board_y) in self.obstacles:
                        grid[gy, gx] = 5
                    # check snake
                    elif (board_x, board_y) in self.snake:
                        grid[gy, gx] = 1
                    else:
                        # check if any food
                        found_food = False
                        for (fx, fy, fcolor, freward) in self.food_items:
                            if (fx, fy) == (board_x, board_y):
                                if fcolor == 'red':
                                    grid[gy, gx] = 2
                                else:  # 'blue'
                                    grid[gy, gx] = 4
                                found_food = True
                                break
                        if not found_food:
                            grid[gy, gx] = 0
        return grid

# ----------------------------------------------------------------
# CNN-based DQN Model
# ----------------------------------------------------------------
class CNNDQN(nn.Module):
    """
    Convolutional network for local grids, shape => (FRAME_STACK, VISION_SIZE, VISION_SIZE).
    Output => Q-values for 3 discrete actions (straight, turn right, turn left).
    """
    def __init__(self, in_channels=FRAME_STACK, num_actions=3):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        conv_out_size = 64 * VISION_SIZE * VISION_SIZE
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

# ----------------------------------------------------------------
# Agent (CNN DQN)
# ----------------------------------------------------------------
class Agent:
    def __init__(self):
        self.action_size = 3  # [straight, turn right, turn left]

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.9
        self.learning_rate = 0.001

        self.memory = deque(maxlen=50000)
        self.batch_size = 32

        self.model = CNNDQN(in_channels=FRAME_STACK, num_actions=self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.model(st)  # shape => (1, 3)
            return torch.argmax(q_vals, dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=device)

        current_Q = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_Q = self.model(next_states_t).max(1)[0]

        target_Q = rewards_t + (self.gamma * max_next_Q * (~dones_t))

        loss = self.criterion(current_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ----------------------------------------------------------------
# Main Training Loop with Enhanced UI & Continuous Food
# ----------------------------------------------------------------
def train_snake(num_episodes=100):
    pygame.init()
    font = get_font(18)
    font_bold = get_font(36, bold=True)
    clock = pygame.time.Clock()

    env = SnakeGameEnv()
    agent = Agent()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake RL - Continuous Food")

    scores = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return scores

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            episode_reward += reward

            # ---------------------------
            # Draw the game UI
            # ---------------------------
            screen.fill(COLOR_BG)

            # Score bar at top
            SCOREBAR_HEIGHT = 40
            pygame.draw.rect(screen, COLOR_SCOREBAR, (0, 0, WIDTH, SCOREBAR_HEIGHT))

            # Draw snake
            for i, (sx, sy) in enumerate(env.snake):
                if i == 0:
                    pygame.draw.rect(screen, COLOR_SNAKE_HEAD, (sx, sy, BLOCK_SIZE, BLOCK_SIZE))
                else:
                    pygame.draw.rect(screen, COLOR_SNAKE_BODY, (sx, sy, BLOCK_SIZE, BLOCK_SIZE))

            # Draw obstacles
            for ox, oy in env.obstacles:
                pygame.draw.rect(screen, COLOR_OBSTACLE, (ox, oy, BLOCK_SIZE, BLOCK_SIZE))

            # Draw foods (red & blue)
            for (fx, fy, fcolor, freward) in env.food_items:
                if fcolor == 'red':
                    pygame.draw.rect(screen, COLOR_RED_FOOD, (fx, fy, BLOCK_SIZE, BLOCK_SIZE))
                else:  # 'blue'
                    pygame.draw.rect(screen, COLOR_BLUE_FOOD, (fx, fy, BLOCK_SIZE, BLOCK_SIZE))

            # Show Score/Eps
            text_surf = font.render(
                f"Episode: {episode+1}/{num_episodes} | Score: {env.score} | Epsilon: {agent.epsilon:.2f}",
                True, COLOR_TEXT
            )
            screen.blit(text_surf, (10, 10))

            # Optionally draw debug overlay of local vision
            if show_vision_overlay:
                overlay = env.grid_stack[-1]  # the most recent vision grid
                # Draw it in bottom-right corner, scaling each cell
                overlay_size = 5  # pixel size per cell in the overlay
                ox_start = WIDTH - (VISION_SIZE * overlay_size) - 10
                oy_start = HEIGHT - (VISION_SIZE * overlay_size) - 10

                for gy in range(VISION_SIZE):
                    for gx in range(VISION_SIZE):
                        val = overlay[gy, gx]
                        color = COLOR_EMPTY
                        if val == 1:
                            color = COLOR_SNAKE_BODY
                        elif val == 2:
                            color = COLOR_RED_FOOD
                        elif val == 3:
                            color = COLOR_WALL
                        elif val == 4:
                            color = COLOR_BLUE_FOOD
                        elif val == 5:
                            color = COLOR_OBSTACLE
                        rect_x = ox_start + gx * overlay_size
                        rect_y = oy_start + gy * overlay_size
                        pygame.draw.rect(screen, color, (rect_x, rect_y, overlay_size, overlay_size))

            pygame.display.flip()

            # Increase speed with score
            speed = 10 + env.score
            clock.tick(speed)

        scores.append(env.score)

        # Show "Game Over" briefly
        game_over_text = font_bold.render(f"GAME OVER! Score: {env.score}", True, COLOR_GAMEOVER)
        screen.blit(game_over_text, (WIDTH // 2 - 140, HEIGHT // 2 - 20))
        pygame.display.flip()
        pygame.time.delay(1000)

        print(f"Episode {episode+1}/{num_episodes}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

    pygame.quit()
    return scores

# ----------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------
if __name__ == "__main__":
    final_scores = train_snake(num_episodes=1000)
    print("Training finished. Scores:", final_scores)
