# Reinforcement Learning for Snake Game

This repository contains multiple implementations of a **Reinforcement Learning-based Snake game**. The implementations utilize **Deep Q-Networks (DQN)**, **CNN-based models**, and **prioritized experience replay** to train an AI to play the game autonomously. The environment is built using **Pygame**, and the models are developed with **PyTorch**.

## 📂 Files in This Repository

### 1. `latest_snake.py`
This script implements a **deep reinforcement learning-based Snake game**. It includes:
- 🐍 A Snake environment with obstacles and multiple food types.
- 📌 Prioritized Experience Replay (PER) for improved learning efficiency.
- 🏆 A **Dueling Deep Q-Network (DQN) with LSTM and Spatial Attention**.
- 🎯 **Reward shaping mechanisms** for better agent learning.

### 2. `new_exp.py`
This file is another experiment-based implementation of the Snake game with:
- 📷 **Frame stacking and local vision grid** for better spatial awareness.
- 🧠 **CNN-based Q-learning model** to predict optimal movements.
- 🍎 **Dynamic food placement** and obstacle handling.

### 3. `new_exp_env_st.py`
This script introduces:
- 🚧 A **Snake game environment with fixed obstacles** across all episodes.
- 📈 **Enhanced training stability** by keeping obstacle positions constant.
- 🎯 **CNN-based Q-learning models** for training agents with better generalization.

### 4. `seek_RL_game.py`
This script is a simplified version of the RL Snake game, focusing on:
- 🏗 **Basic Q-learning with a small neural network**.
- 📊 **State representation including obstacles and food locations**.
- 🎲 **Epsilon-greedy action selection** for training.

### 5. `snake_autoplay.py`
This script automates gameplay for the Snake AI with:
- 🤖 **A simple RL model using deep Q-learning**.
- 🎯 **Direction-based decision-making for snake movement**.
- 🍏 **Food tracking for improved pathfinding**.

### 6. `update_snake_gpt.py`
This script improves upon previous implementations by:
- 🔍 **Enhancing the vision grid** for the snake to make more informed decisions.
- 🏆 **Using CNN-based Q-learning** for training a more robust AI model.
- 🧠 **Implementing frame stacking** for better temporal awareness in decision-making.

## 🚀 How to Run the Scripts

1. Install dependencies:
   ```bash
   pip install pygame numpy torch
   ```
2. Run any script using:
   ```bash
   python latest_snake.py
   ```
   _(Replace `latest_snake.py` with the script you want to run.)_

## 🔮 Future Improvements

- 🚀 **Implementing PPO (Proximal Policy Optimization)** for better training efficiency.
- 🏗 **Testing different neural network architectures** for performance comparison.
- 🎨 **Integrating a GUI interface** for better visualization of AI learning progress.

---

Feel free to contribute or raise issues if you encounter any problems! 🚀

