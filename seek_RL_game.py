import pygame
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define directions
class Direction:
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    UP = (0, -1)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class SnakeGame:
    def __init__(self, w=20, h=20, render=False):
        self.w = w
        self.h = h
        self.render = render
        self.block_size = 20
        self.speed = 10
        self.directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        
        if self.render:
            pygame.init()
            self.font = pygame.font.Font(None, 30)
            self.display = pygame.display.set_mode((self.w * self.block_size, 
                                                   self.h * self.block_size))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head,
                      Point(self.head.x-1, self.head.y),
                      Point(self.head.x-2, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
    
    def _place_food(self):
        while True:
            x = random.randint(0, self.w-1)
            y = random.randint(0, self.h-1)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        if pt in self.snake[1:]:
            return True
        return False
    
    def get_state(self):
        head = self.head
        current_idx = self.directions.index(self.direction)
        
        dir_straight = self.direction
        dir_right = self.directions[(current_idx + 1) % 4]
        dir_left = self.directions[(current_idx - 1) % 4]
        
        danger = [
            self.is_collision(Point(head.x + dir_straight[0], head.y + dir_straight[1])),
            self.is_collision(Point(head.x + dir_right[0], head.y + dir_right[1])),
            self.is_collision(Point(head.x + dir_left[0], head.y + dir_left[1]))
        ]
        
        dir_one_hot = [0]*4
        dir_one_hot[current_idx] = 1
        
        food_rel_x = self.food.x - head.x
        food_rel_y = self.food.y - head.y
        
        state = [
            # Danger directions
            danger[0], danger[1], danger[2],
            
            # Current direction
            *dir_one_hot,
            
            # Food location
            food_rel_x < 0,  # food left
            food_rel_x > 0,  # food right
            food_rel_y < 0,  # food up
            food_rel_y > 0   # food down
        ]
        
        return np.array(state, dtype=int)
    
    def play_step(self, action):
        self.frame_iteration += 1
        
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        current_idx = self.directions.index(self.direction)
        if action == 1:  # Right turn
            new_dir = self.directions[(current_idx + 1) % 4]
        elif action == 2:  # Left turn
            new_dir = self.directions[(current_idx - 1) % 4]
        else:  # Straight
            new_dir = self.direction
        
        self.direction = new_dir
        new_head = Point(self.head.x + self.direction[0], 
                        self.head.y + self.direction[1])
        
        game_over = False
        reward = 0
        
        if self.is_collision(new_head):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        self.snake.insert(0, new_head)
        self.head = new_head
        
        if self.head == self.food:
            self.score += 1
            self.speed += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1
        
        if self.render:
            self.display.fill((0, 0, 0))
            for pt in self.snake:
                pygame.draw.rect(self.display, (0, 255, 0), 
                                pygame.Rect(pt.x * self.block_size, pt.y * self.block_size,
                                            self.block_size, self.block_size))
            pygame.draw.rect(self.display, (255, 0, 0), 
                            pygame.Rect(self.food.x * self.block_size, self.food.y * self.block_size,
                                        self.block_size, self.block_size))
            text = self.font.render(f"Score: {self.score} Speed: {self.speed}", True, (255, 255, 255))
            self.display.blit(text, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.speed)
        
        return reward, game_over, self.score

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Agent:
    def __init__(self, input_size, hidden_size, output_size):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        
        self.model = DQN(input_size, hidden_size, output_size)
        self.target_model = DQN(input_size, hidden_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

def train():
    game = SnakeGame(render=True)  # Set render=False for faster training
    agent = Agent(input_size=11, hidden_size=256, output_size=3)
    episodes = 1000
    update_target_every = 50
    
    for episode in range(episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0
        
        while True:
            action = agent.get_action(state)
            reward, done, score = game.play_step(action)
            next_state = game.get_state()  # Always get the next state
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if episode % update_target_every == 0:
            agent.update_target_model()
        
        print(f'Episode {episode}, Score: {score}, Epsilon: {agent.epsilon:.2f}')
    
    torch.save(agent.model.state_dict(), 'snake_dqn.pth')

if __name__ == '__main__':
    train()