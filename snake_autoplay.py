import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------
# Global Constants
# ------------------------------------
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20

# By default, PyTorch will run on CPU unless we specify otherwise.
device = torch.device("cpu")  # Force CPU usage.

# -----------------------------
# Helper Functions
# -----------------------------
def collide(x, y):
    """Check if (x, y) is outside the screen boundary."""
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return True
    return False

def get_font(size=20):
    """Helper to load a system font. Fallbacks to default if not found."""
    return pygame.font.SysFont("arial", size)

# -----------------------------
# Snake Game Environment
# -----------------------------
class SnakeGameEnv:
    """
    This environment manages the snake, food, rewards, collisions.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the game for a new episode.
        Returns the initial state.
        """
        self.direction = 0  # 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        self.score = 0
        self.game_over = False

        # Snake starting at the center
        mid_x = WIDTH // 2
        mid_y = HEIGHT // 2

        # Initial snake: 3 blocks
        self.snake = [
            (mid_x, mid_y),
            (mid_x - BLOCK_SIZE, mid_y),
            (mid_x - 2 * BLOCK_SIZE, mid_y)
        ]
        self.head = self.snake[0]

        # Place first food
        self._place_food()

        # Count frames for potential additional logic
        self.frame_iteration = 0

        # Return the initial state
        return self.get_state()

    def _place_food(self):
        """Randomly place food on the grid (not on the snake)."""
        x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = (x, y)
        # In case food lands where the snake is
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        """
        action: 0 = straight, 1 = turn right, 2 = turn left
        Returns: next_state, reward, done
        """
        self.frame_iteration += 1

        # 1. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 2. Check collision
        reward = 0
        if collide(self.head[0], self.head[1]) or (self.head in self.snake[1:]):
            # Game over
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True

        # 3. Check if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 4. Prepare next state
        next_state = self.get_state()
        done = False
        return next_state, reward, done

    def _move(self, action):
        """
        Update the direction based on the given action, then update the head position.
        direction: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
        action: 0=straight, 1=right, 2=left
        """
        clock_wise = [0, 1, 2, 3]
        idx = clock_wise.index(self.direction)

        if action == 1:
            # turn right -> idx + 1
            idx = (idx + 1) % 4
        elif action == 2:
            # turn left -> idx - 1
            idx = (idx - 1) % 4

        self.direction = clock_wise[idx]

        x, y = self.head
        if self.direction == 0:  # RIGHT
            x += BLOCK_SIZE
        elif self.direction == 1:  # DOWN
            y += BLOCK_SIZE
        elif self.direction == 2:  # LEFT
            x -= BLOCK_SIZE
        elif self.direction == 3:  # UP
            y -= BLOCK_SIZE

        self.head = (x, y)

    def get_state(self):
        """
        Construct the state representation as a numpy array.
        We'll encode:
          1. Danger (straight, left, right)
          2. Direction (up, right, down, left)
          3. Food location (up, right, down, left)
        """
        head_x, head_y = self.head

        # Identify left and right directions (relative to current direction)
        left_dir = (self.direction - 1) % 4
        right_dir = (self.direction + 1) % 4

        # Coordinates if we move straight, left, or right
        front_block = self._get_next_block(head_x, head_y, self.direction)
        left_block = self._get_next_block(head_x, head_y, left_dir)
        right_block = self._get_next_block(head_x, head_y, right_dir)

        # Danger check
        danger_straight = (collide(front_block[0], front_block[1]) or front_block in self.snake)
        danger_left = (collide(left_block[0], left_block[1]) or left_block in self.snake)
        danger_right = (collide(right_block[0], right_block[1]) or right_block in self.snake)

        # Direction encoding (boolean)
        dir_up = (self.direction == 3)
        dir_right = (self.direction == 0)
        dir_down = (self.direction == 1)
        dir_left = (self.direction == 2)

        # Food location relative to head
        food_x, food_y = self.food
        food_up = (food_y < head_y)
        food_down = (food_y > head_y)
        food_left = (food_x < head_x)
        food_right = (food_x > head_x)

        state = [
            # Danger
            int(danger_straight),
            int(danger_left),
            int(danger_right),

            # Direction
            int(dir_up),
            int(dir_right),
            int(dir_down),
            int(dir_left),

            # Food location
            int(food_up),
            int(food_right),
            int(food_down),
            int(food_left),
        ]

        return np.array(state, dtype=np.float32)

    def _get_next_block(self, x, y, direction):
        """
        Helper to get the next block if moving in the specified direction.
        """
        if direction == 0:  # RIGHT
            x += BLOCK_SIZE
        elif direction == 1:  # DOWN
            y += BLOCK_SIZE
        elif direction == 2:  # LEFT
            x -= BLOCK_SIZE
        elif direction == 3:  # UP
            y -= BLOCK_SIZE
        return (x, y)

# -----------------------------
# DQN Model
# -----------------------------
class DQN(nn.Module):
    """
    A simple fully-connected neural network for approximating Q-values.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# -----------------------------
# Agent using DQN
# -----------------------------
class Agent:
    """
    The RL agent that learns via a Deep Q-Network.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Exploration parameters
        self.epsilon = 1.0       # initial exploration rate
        self.epsilon_min = 0.01  # minimal exploration rate
        self.epsilon_decay = 0.995

        # Discount factor
        self.gamma = 0.9

        # Learning rate for optimizer
        self.learning_rate = 0.001

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 64

        # Create DQN
        self.model = DQN(state_size, 128, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        """
        Select an action using epsilon-greedy strategy.
        """
        # Exploration
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation
        state_t = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Sample a mini-batch from memory and train the DQN.
        """
        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # Current Q values
        current_Q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values
        with torch.no_grad():
            max_next_Q = self.model(next_states).max(1)[0]
        target_Q = rewards + (self.gamma * max_next_Q * (~dones))

        loss = self.criterion(current_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -----------------------------
# Main Training Function
# -----------------------------
def train_snake(num_episodes=300):
    """
    num_episodes: Number of games (episodes) to train the snake.
    """
    pygame.init()
    font = get_font(20)
    clock = pygame.time.Clock()

    # Initialize environment and agent
    env = SnakeGameEnv()
    state_size = 11   # We have an 11-element state
    action_size = 3   # [straight, turn right, turn left]
    agent = Agent(state_size, action_size)

    # For display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Autonomous Snake (DQN)")

    # To store scores for analysis
    scores = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            # Handle PyGame events (quit, etc.)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return scores

            # Agent picks an action
            action = agent.get_action(state)

            # Step in the environment
            next_state, reward, done = env.step(action)

            # Save experience
            agent.remember(state, action, reward, next_state, done)
            # Train the agent from replay buffer
            agent.replay()

            # Update state and episode reward
            state = next_state
            episode_reward += reward

            # Render the game
            screen.fill((0, 0, 0))

            # Draw snake
            for x, y in env.snake:
                pygame.draw.rect(screen, (0, 255, 0), (x, y, BLOCK_SIZE, BLOCK_SIZE))

            # Draw food
            fx, fy = env.food
            pygame.draw.rect(screen, (255, 0, 0), (fx, fy, BLOCK_SIZE, BLOCK_SIZE))

            # Display stats
            text = font.render(f"Episode: {episode}   Score: {env.score}   Epsilon: {agent.epsilon:.2f}", True, (255, 255, 255))
            screen.blit(text, [10, 10])

            pygame.display.flip()

            # Increase speed with score
            speed = 10 + env.score
            clock.tick(speed)

            if env.game_over:
                scores.append(env.score)
                break

        print(f"Episode {episode+1}/{num_episodes}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

    pygame.quit()
    return scores

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    final_scores = train_snake(num_episodes=300)
    print("Training completed!")
    print("Scores per episode:", final_scores)
