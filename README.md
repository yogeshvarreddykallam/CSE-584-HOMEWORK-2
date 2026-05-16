# Q-Learning Tic-Tac-Toe

A **Q-learning reinforcement learning agent** trained to play Tic-Tac-Toe from scratch using an epsilon-greedy exploration policy.

Submitted as Homework 2 for **PSU CSE 584 — Machine Learning**.

## Overview

The agent starts with zero knowledge of the game and learns purely through self-play. Over thousands of episodes it converges toward an optimal strategy by iteratively updating a Q-table using the **Bellman equation**:

```
Q(s,a) = Q(s,a) + α [ r + γ · max Q(s',a') − Q(s,a) ]
```

Where:
- `s` = current board state (9-element list: empty / X / O)
- `a` = chosen action (cell index 0–8)
- `r` = reward (+1 win, +0.5 draw, −1 loss)
- `α` = learning rate
- `γ` = discount factor

## How It Works

1. **Initialization** — Q-table set to zero for all (state, action) pairs
2. **Training loop** — agent plays *N* episodes against a random opponent:
   - Picks an action via **ε-greedy** policy (explore or exploit)
   - Receives a reward after each move
   - Updates Q-values with the Bellman equation
   - ε decays over time, shifting from exploration → exploitation
3. **Evaluation** — trained agent plays against a human or random opponent; win rate improves as Q-values converge

## Key Components

| Component | Description |
|-----------|-------------|
| Q-table | Dictionary mapping (state, action) → expected reward |
| ε-greedy policy | High ε early (random exploration) → low ε later (exploit learned values) |
| Reward shaping | Win: +1 · Draw: +0.5 · Loss: −1 |
| Bellman update | Off-policy TD update using best future Q-value |

## Hyperparameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| Learning rate | α | How fast new info overwrites old Q-values |
| Discount factor | γ | Weight of future rewards vs. immediate reward |
| Epsilon start | ε₀ | Initial exploration probability |
| Epsilon decay | δ | Rate at which ε shrinks each episode |

## Usage

```bash
pip install numpy

python hw2.py              # Train and evaluate
```

## Report

Full writeup with line-by-line code commentary: [`CSE 584, HOMEWORK 2.pdf`](CSE%20584%2C%20HOMEWORK%202.pdf)
