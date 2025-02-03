import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------------------------
# Global Constants
# ------------------------------------
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20

VISION_SIZE = 11  # local vision grid side
FRAME_STACK = 4   # how many consecutive frames we stack

device = torch.device("cpu")  # Force CPU usage

# Colors (R,G,B)
COLOR_BG = (30, 30, 30)           # dark background
COLOR_SCOREBAR = (60, 60, 60)     # top bar
COLOR_SNAKE_BODY = (0, 180, 0)
COLOR_SNAKE_HEAD = (0, 255, 0)
COLOR_FOOD = (255, 50, 50)
COLOR_BORDER = (100, 100, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_GAMEOVER = (200, 50, 50)

def get_font(size=20, bold=False):
    """Helper to load a system font, fallback to default if not found."""
    return pygame.font.SysFont("arial", size, bold=bold)

# -----------------------------
# Snake Game Environment
# -----------------------------
class SnakeGameEnv:
    """
    Environment with:
      - Snake's movement
      - Food placement
      - Local vision grid
      - Frame stacking
      - Enhanced visuals come from the main loop (outside).
    """
    def __init__(self):
        self.direction = 0  # 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        self.snake = None
        self.head = None
        self.food = None
        self.score = 0
        self.game_over = False

        # Store last N=FRAME_STACK vision grids
        self.grid_stack = deque(maxlen=FRAME_STACK)

        self.reset()

    def reset(self):
        """Reset environment for a new episode."""
        self.direction = 0
        self.score = 0
        self.game_over = False

        # Center of screen
        mid_x = WIDTH // 2
        mid_y = HEIGHT // 2

        # Start snake horizontally, length = 3
        self.snake = [
            (mid_x, mid_y),
            (mid_x - BLOCK_SIZE, mid_y),
            (mid_x - 2 * BLOCK_SIZE, mid_y),
        ]
        self.head = self.snake[0]

        # Place initial food
        self._place_food()

        # Initialize frame stack with empty grids
        self.grid_stack.clear()
        empty_grid = np.zeros((VISION_SIZE, VISION_SIZE), dtype=np.float32)
        for _ in range(FRAME_STACK):
            self.grid_stack.append(empty_grid)

        return self.get_state()

    def _place_food(self):
        """Place food randomly, not on snake body."""
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake:
                self.food = (x, y)
                return

    def step(self, action):
        """
        action: 0=straight, 1=turn right, 2=turn left
        returns: (next_state, reward, done)
        """
        # 1. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 2. Check collision
        reward = 0
        if self._collided(self.head[0], self.head[1]) or (self.head in self.snake[1:]):
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True

        # 3. Check food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        next_state = self.get_state()
        done = False
        return next_state, reward, done

    def _collided(self, x, y):
        """Check if outside board boundaries."""
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return False

    def _move(self, action):
        """
        Update direction according to action, then move head one block.
        direction: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        """
        directions = [0, 1, 2, 3]
        idx = directions.index(self.direction)

        if action == 1:  # turn right
            idx = (idx + 1) % 4
        elif action == 2:  # turn left
            idx = (idx - 1) % 4

        self.direction = directions[idx]

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
        Return stacked frames of local vision grids, shape = (FRAME_STACK, VISION_SIZE, VISION_SIZE).
        """
        grid = self._get_vision_grid()
        self.grid_stack.append(grid)
        stacked_state = np.stack(self.grid_stack, axis=0)  # shape (4, 11, 11)
        return stacked_state

    def _get_vision_grid(self):
        """
        Create a local VISION_SIZE x VISION_SIZE grid around the snake's head.
        Encoding:
         0 = empty
         1 = snake body
         2 = food
         3 = wall/out-of-bounds
        """
        grid = np.zeros((VISION_SIZE, VISION_SIZE), dtype=np.float32)
        radius = VISION_SIZE // 2

        head_x, head_y = self.head

        for cx in range(-radius, radius + 1):
            for cy in range(-radius, radius + 1):
                gx = cx + radius
                gy = cy + radius
                board_x = head_x + cx * BLOCK_SIZE
                board_y = head_y + cy * BLOCK_SIZE

                if (board_x < 0 or board_x >= WIDTH or
                    board_y < 0 or board_y >= HEIGHT):
                    # Wall
                    grid[gy, gx] = 3
                else:
                    if (board_x, board_y) in self.snake:
                        grid[gy, gx] = 1  # snake
                    elif (board_x, board_y) == self.food:
                        grid[gy, gx] = 2  # food
                    else:
                        grid[gy, gx] = 0
        return grid

# -----------------------------
# CNN-based DQN Model
# -----------------------------
class CNNDQN(nn.Module):
    """
    Convolutional network to process stacked grid states (FRAME_STACK x VISION_SIZE x VISION_SIZE).
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

# -----------------------------
# Agent (CNN DQN)
# -----------------------------
class Agent:
    def __init__(self):
        self.action_size = 3  # straight, right, left

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.9
        self.learning_rate = 0.001

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.model = CNNDQN(in_channels=FRAME_STACK, num_actions=self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.model(st)
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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -----------------------------
# Main Training with Enhanced UI
# -----------------------------
def train_snake(num_episodes=100):
    pygame.init()
    font = get_font(18)
    font_bold = get_font(40, bold=True)
    clock = pygame.time.Clock()

    env = SnakeGameEnv()
    agent = Agent()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake with Enhanced UI (CNN + Vision)")

    scores = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Handle close event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return scores

            # Agent picks an action
            action = agent.get_action(state)

            # Step
            next_state, reward, done = env.step(action)

            # Store and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            episode_reward += reward

            # ---------------------------
            # Enhanced UI Drawing
            # ---------------------------
            screen.fill(COLOR_BG)

            # Draw the scoreboard background at the top
            SCOREBAR_HEIGHT = 40
            pygame.draw.rect(screen, COLOR_SCOREBAR, (0, 0, WIDTH, SCOREBAR_HEIGHT))

            # Draw a thin border around the playable area
            # (just below the scoreboard)
            border_rect = pygame.Rect(0, SCOREBAR_HEIGHT, WIDTH, HEIGHT - SCOREBAR_HEIGHT)
            pygame.draw.rect(screen, COLOR_BORDER, border_rect, 2)

            # Draw snake
            for i, (sx, sy) in enumerate(env.snake):
                if i == 0:  # head
                    pygame.draw.rect(screen, COLOR_SNAKE_HEAD, (sx, sy, BLOCK_SIZE, BLOCK_SIZE))
                else:
                    pygame.draw.rect(screen, COLOR_SNAKE_BODY, (sx, sy, BLOCK_SIZE, BLOCK_SIZE))

            # Draw food
            fx, fy = env.food
            pygame.draw.rect(screen, COLOR_FOOD, (fx, fy, BLOCK_SIZE, BLOCK_SIZE))

            # Render Score/Epsilon on scoreboard
            text_surf = font.render(
                f"Episode: {episode+1}/{num_episodes} | Score: {env.score} | Epsilon: {agent.epsilon:.2f}",
                True, COLOR_TEXT
            )
            screen.blit(text_surf, (10, 10))

            pygame.display.flip()

            # Increase speed with score
            speed = 10 + env.score
            clock.tick(speed)

        scores.append(env.score)

        # ---------------------------
        # Game Over Screen (brief)
        # ---------------------------
        game_over_text = font_bold.render(f"GAME OVER! Score: {env.score}", True, COLOR_GAMEOVER)
        screen.blit(game_over_text, (WIDTH // 2 - 150, HEIGHT // 2 - 20))
        pygame.display.flip()

        # Pause for a second to show "Game Over"
        pygame.time.delay(1000)

        print(f"Episode {episode+1}/{num_episodes}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

    pygame.quit()
    return scores


if __name__ == "__main__":
    final_scores = train_snake(num_episodes=50)
    print("Training completed. Scores:", final_scores)
