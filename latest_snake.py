import pygame
import random
import math
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----------------------------------------------------------------
# Global Parameters
# ----------------------------------------------------------------
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20

VISION_SIZE = 11    # local vision grid (must be odd)
FRAME_STACK = 4     # consecutive frames for LSTM or stacked input

NUM_RED_FOODS = 1
NUM_BLUE_FOODS = 1
NUM_OBSTACLES = 5

show_vision_overlay = True

device = torch.device("cpu")  # or torch.device("cuda") if you have a GPU

# Reward shaping parameters
DISTANCE_REWARD_SCALE = 0.1     # how strongly we reward moving closer to food
SURVIVAL_BONUS = 0.01           # small reward each step for staying alive
HUNGER_STEP_LIMIT = 200         # steps allowed without eating before a penalty
HUNGER_PENALTY = -5.0           # penalty if we exceed hunger limit

# ----------------------------------------------------------------
# Colors
# ----------------------------------------------------------------
COLOR_BG = (30, 30, 30)
COLOR_SNAKE_BODY = (0, 180, 0)
COLOR_SNAKE_HEAD = (0, 255, 0)
COLOR_RED_FOOD = (255, 50, 50)   # reward 10
COLOR_BLUE_FOOD = (50, 50, 255)  # reward 15
COLOR_OBSTACLE = (100, 100, 100)
COLOR_WALL = (120, 120, 120)
COLOR_EMPTY = (200, 200, 200)
COLOR_SCOREBAR = (60, 60, 60)
COLOR_TEXT = (255, 255, 255)
COLOR_GAMEOVER = (200, 50, 50)

def get_font(size=20, bold=False):
    return pygame.font.SysFont("arial", size, bold=bold)

# ----------------------------------------------------------------
# Prioritized Replay Buffer
# ----------------------------------------------------------------
class PrioritizedReplayBuffer:
    """
    Stores transitions with a priority value.
    Higher priority => more likely to be sampled.
    """
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.epsilon = 1e-5  # small constant to avoid zero priority

    def add(self, transition, priority=1.0):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if max_prio == 0:
            max_prio = priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=32, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return indices, samples, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# ----------------------------------------------------------------
# Snake Environment (Fixed obstacles, continuous food)
# ----------------------------------------------------------------
class SnakeGameEnv:
    def __init__(self):
        self.direction = 0  # 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        self.snake = None
        self.head = None
        self.score = 0
        self.game_over = False

        self.food_items = []
        self.obstacles = set()
        self.grid_stack = deque(maxlen=FRAME_STACK)

        self._place_obstacles_once()

        # Tracking steps since last food eaten (for hunger penalty)
        self.steps_since_last_food = 0

        self.reset()

    def _place_obstacles_once(self):
        for _ in range(NUM_OBSTACLES):
            while True:
                x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                if (x, y) not in self.obstacles:
                    self.obstacles.add((x, y))
                    break

    def reset(self):
        self.direction = 0
        self.score = 0
        self.game_over = False

        # Center snake
        mid_x = WIDTH // 2
        mid_y = HEIGHT // 2
        self.snake = [
            (mid_x, mid_y),
            (mid_x - BLOCK_SIZE, mid_y),
            (mid_x - 2 * BLOCK_SIZE, mid_y),
        ]
        self.head = self.snake[0]

        self.food_items = []
        self._place_foods()

        self.grid_stack.clear()
        # IMPORTANT: each frame must be shape (2, VISION_SIZE, VISION_SIZE)
        blank_frame = np.zeros((2, VISION_SIZE, VISION_SIZE), dtype=np.float32)
        for _ in range(FRAME_STACK):
            self.grid_stack.append(blank_frame)

        self.steps_since_last_food = 0

        return self.get_state()

    def _place_foods(self):
        for _ in range(NUM_RED_FOODS):
            self._spawn_food(reward=10, color='red')
        for _ in range(NUM_BLUE_FOODS):
            self._spawn_food(reward=15, color='blue')

    def _spawn_food(self, reward, color):
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake and (x, y) not in self.obstacles:
                self.food_items.append((x, y, color, reward))
                return

    def step(self, action):
        # Move
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        done = False

        # Check collisions
        if self._collided_wall(self.head[0], self.head[1]) or (self.head in self.snake[1:]):
            reward = -10
            self.game_over = True
            done = True
            return self.get_state(), reward, done

        if self.head in self.obstacles:
            reward = -10
            self.game_over = True
            done = True
            return self.get_state(), reward, done

        # Check food
        eaten_idx = -1
        for i, (fx, fy, fcolor, freward) in enumerate(self.food_items):
            if (fx, fy) == self.head:
                eaten_idx = i
                self.score += 1
                reward += freward
                self.steps_since_last_food = 0
                break

        if eaten_idx >= 0:
            x_old, y_old, color_old, reward_old = self.food_items[eaten_idx]
            self.food_items[eaten_idx] = self._random_new_food(color_old, reward_old)
        else:
            self.snake.pop()

        # Survival bonus
        reward += SURVIVAL_BONUS

        # Distance-based reward
        nearest_food_dist = self._nearest_food_distance(self.head[0], self.head[1])
        if nearest_food_dist is not None:
            inv_dist = 1.0 / (nearest_food_dist + 1.0)
            reward += DISTANCE_REWARD_SCALE * inv_dist

        # Hunger penalty
        self.steps_since_last_food += 1
        if self.steps_since_last_food > HUNGER_STEP_LIMIT:
            reward += HUNGER_PENALTY
            self.game_over = True
            done = True

        next_state = self.get_state()
        return next_state, reward, done

    def _random_new_food(self, color, reward):
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake and (x, y) not in self.obstacles:
                return (x, y, color, reward)

    def _collided_wall(self, x, y):
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return False

    def _move(self, action):
        # directions: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        dirs = [0, 1, 2, 3]
        idx = dirs.index(self.direction)
        if action == 1:  # turn right
            idx = (idx + 1) % 4
        elif action == 2:  # turn left
            idx = (idx - 1) % 4

        self.direction = dirs[idx]

        x, y = self.head
        if self.direction == 0:  # RIGHT
            x += BLOCK_SIZE
        elif self.direction == 1: # DOWN
            y += BLOCK_SIZE
        elif self.direction == 2: # LEFT
            x -= BLOCK_SIZE
        elif self.direction == 3: # UP
            y -= BLOCK_SIZE

        self.head = (x, y)

    def _nearest_food_distance(self, x, y):
        if not self.food_items:
            return None
        dists = []
        for (fx, fy, _, _) in self.food_items:
            dx = (fx - x) / BLOCK_SIZE
            dy = (fy - y) / BLOCK_SIZE
            dist = abs(dx) + abs(dy)  # Manhattan distance
            dists.append(dist)
        return min(dists) if dists else None

    def get_state(self):
        # Construct the vision grid
        grid = self._get_vision_grid()

        # Extra channel for direction
        dir_channel = np.full((VISION_SIZE, VISION_SIZE), self.direction, dtype=np.float32)
        combined_grid = np.stack([grid, dir_channel], axis=0)  # shape => (2, VISION_SIZE, VISION_SIZE)

        # Add to stack
        self.grid_stack.append(combined_grid)

        # If self.grid_stack not yet full, prepend blank frames
        frames = list(self.grid_stack)
        while len(frames) < FRAME_STACK:
            blank_frame = np.zeros((2, VISION_SIZE, VISION_SIZE), dtype=np.float32)
            frames.insert(0, blank_frame)

        # shape => (2*FRAME_STACK, VISION_SIZE, VISION_SIZE)
        stacked_state = np.concatenate(frames, axis=0)
        return stacked_state

    def _get_vision_grid(self):
        """
        0 = empty
        1 = snake
        2 = red food
        3 = wall
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
                    grid[gy, gx] = 3
                else:
                    if (board_x, board_y) in self.obstacles:
                        grid[gy, gx] = 5
                    elif (board_x, board_y) in self.snake:
                        grid[gy, gx] = 1
                    else:
                        found_food = False
                        for (fx, fy, fcolor, freward) in self.food_items:
                            if (fx, fy) == (board_x, board_y):
                                if fcolor == 'red':
                                    grid[gy, gx] = 2
                                else:
                                    grid[gy, gx] = 4
                                found_food = True
                                break
                        if not found_food:
                            grid[gy, gx] = 0
        return grid

# ----------------------------------------------------------------
# Spatial Attention Layer
# ----------------------------------------------------------------
class AttentionLayer(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x shape => (batch, channels, H, W)
        attn_map = self.conv1(x)  # => (batch, 1, H, W)
        attn_weights = torch.sigmoid(attn_map)  # 0..1
        return x * attn_weights

# ----------------------------------------------------------------
# Dueling + LSTM-based DQN with Attention
# ----------------------------------------------------------------
class DuelingLSTMDQN(nn.Module):
    def __init__(self, in_channels=2*FRAME_STACK, hidden_size=128, height=VISION_SIZE, width=VISION_SIZE, num_actions=3):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.num_actions = num_actions

        # CNN feature extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Spatial Attention
        self.attn = AttentionLayer(channels=64, height=height, width=width)

        # LSTM
        self.lstm_input_size = 64 * height * width
        self.lstm_hidden_size = hidden_size
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, batch_first=True)

        # Dueling heads
        self.value_fc = nn.Linear(hidden_size, 1)
        self.advantage_fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x, lstm_hidden=None):
        # x => (batch, in_channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Attention
        x = self.attn(x)

        # Flatten
        b, c, h, w = x.shape
        x = x.view(b, -1)  # => (b, c*h*w)

        # LSTM expects (batch, seq_len, input_size)
        x = x.unsqueeze(1)  # seq_len=1
        if lstm_hidden is None:
            batch_size = b
            h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
            c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
            lstm_hidden = (h0, c0)

        lstm_out, (h_n, c_n) = self.lstm(x, lstm_hidden)
        feat = lstm_out[:, -1, :]  # => (b, hidden_size)

        # Dueling
        value = self.value_fc(feat)            # => (b, 1)
        advantage = self.advantage_fc(feat)    # => (b, num_actions)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        return q_values, (h_n, c_n)

# ----------------------------------------------------------------
# Agent with Double DQN + Boltzmann Exploration + PER
# ----------------------------------------------------------------
class Agent:
    def __init__(self, in_channels=2*FRAME_STACK, num_actions=3):
        self.num_actions = num_actions

        # Online + Target nets
        self.online_net = DuelingLSTMDQN(in_channels=in_channels, num_actions=num_actions).to(device)
        self.target_net = DuelingLSTMDQN(in_channels=in_channels, num_actions=num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # PER
        self.memory = PrioritizedReplayBuffer(capacity=50000)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=0.0005)
        self.gamma = 0.9

        # Boltzmann exploration
        self.temperature = 1.0
        self.temperature_min = 0.01
        self.temperature_decay = 0.995

        self.batch_size = 32
        self.beta = 0.4
        self.beta_increment = 1e-3

        self.lstm_hidden = None  # keep LSTM state if desired

    def reset_lstm(self):
        self.lstm_hidden = None

    def get_action(self, state):
        """Boltzmann (softmax) exploration using Q-values / temperature."""
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values, self.lstm_hidden = self.online_net(state_t, self.lstm_hidden)
        q_values = q_values[0]  # => (num_actions,)

        logits = q_values / self.temperature
        probs = F.softmax(logits, dim=0).cpu().detach().numpy()
        action = np.random.choice(self.num_actions, p=probs)
        return action

    def store(self, transition, priority=1.0):
        self.memory.add(transition, priority)

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.beta = min(1.0, self.beta + self.beta_increment)

        indices, samples, weights = self.memory.sample(self.batch_size, beta=self.beta)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, ns, d in samples:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

        # Online net forward
        q_values, _ = self.online_net(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN
        next_q_online, _ = self.online_net(next_states_t)
        next_actions = torch.argmax(next_q_online, dim=1)

        with torch.no_grad():
            next_q_target, _ = self.target_net(next_states_t)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + (self.gamma * next_q * (~dones_t))

        td_errors = (current_q - target_q).abs().detach().cpu().numpy()

        # Weighted MSE
        loss = ((current_q - target_q) ** 2) * weights_t
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors + self.memory.epsilon)

        # Decay temperature
        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay

# ----------------------------------------------------------------
# Main Training Loop
# ----------------------------------------------------------------
def train_snake(num_episodes=300):
    pygame.init()
    font = get_font(18)
    clock = pygame.time.Clock()

    env = SnakeGameEnv()
    agent = Agent(in_channels=2*FRAME_STACK, num_actions=3)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake RL - Advanced Features (Fixed Dimension)")

    scores = []
    target_update_interval = 50  # how often to update target net

    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_lstm()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            step_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return scores

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.store((state, action, reward, next_state, done))
            agent.replay()

            state = next_state
            total_reward += reward

            # ---------------
            # Render
            # ---------------
            screen.fill(COLOR_BG)
            SCOREBAR_HEIGHT = 40
            pygame.draw.rect(screen, COLOR_SCOREBAR, (0, 0, WIDTH, SCOREBAR_HEIGHT))

            # Snake
            for i, (sx, sy) in enumerate(env.snake):
                if i == 0:
                    pygame.draw.rect(screen, COLOR_SNAKE_HEAD, (sx, sy, BLOCK_SIZE, BLOCK_SIZE))
                else:
                    pygame.draw.rect(screen, COLOR_SNAKE_BODY, (sx, sy, BLOCK_SIZE, BLOCK_SIZE))

            # Obstacles
            for ox, oy in env.obstacles:
                pygame.draw.rect(screen, COLOR_OBSTACLE, (ox, oy, BLOCK_SIZE, BLOCK_SIZE))

            # Foods
            for (fx, fy, fcolor, frew) in env.food_items:
                if fcolor == 'red':
                    pygame.draw.rect(screen, COLOR_RED_FOOD, (fx, fy, BLOCK_SIZE, BLOCK_SIZE))
                else:
                    pygame.draw.rect(screen, COLOR_BLUE_FOOD, (fx, fy, BLOCK_SIZE, BLOCK_SIZE))

            # Info
            info_text = f"Ep:{episode+1}/{num_episodes} Score:{env.score} Temp:{agent.temperature:.2f}"
            text_surf = font.render(info_text, True, COLOR_TEXT)
            screen.blit(text_surf, (10, 10))

            # Vision overlay
            if show_vision_overlay:
                # The final state has shape (2*FRAME_STACK, 11, 11).
                # Show only the first channel of the last appended frame for debugging:
                overlay = state[0, :, :]  # shape => (11, 11)
                overlay_size = 5
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
                        rx = ox_start + gx * overlay_size
                        ry = oy_start + gy * overlay_size
                        pygame.draw.rect(screen, color, (rx, ry, overlay_size, overlay_size))

            pygame.display.flip()
            speed = 10 + env.score
            clock.tick(speed)

        scores.append(env.score)

        # Periodically update target net
        if (episode+1) % target_update_interval == 0:
            agent.update_target()

        print(f"Episode {episode+1}/{num_episodes}, Score={env.score}, Steps={step_count}, "
              f"Temp={agent.temperature:.2f}, BufferSize={len(agent.memory)}")

    pygame.quit()
    return scores

# ----------------------------------------------------------------
# Entry
# ----------------------------------------------------------------
if __name__ == "__main__":
    final_scores = train_snake(num_episodes=2000)
    print("Training Done. Scores:", final_scores)
