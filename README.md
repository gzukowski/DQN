# DQN

Teaching a drone to land softly with reinforcement learning: tabular Q-learning first, then a Deep Q-Network (DQN)
and Double DQN (DDQN).

## Environment

A 1D vertical landing simulation (`src/dqn/simulation`, `src/dqn/environment`).

- **State:** height (0–100 m) and vertical velocity (±20 m/s)
- **Actions:** `FREE_FALL` (no thrust), `CANCEL_GRAVITY` (thrust = g, keeps current velocity), `BRAKE` (thrust = 3g)
- **Start:** height U[50, 100] m, velocity U[−5, 0] m/s
- **Goal:** touch down slower than 2 m/s
- **Rewards:** +100 soft landing, −100 crash, −200 out of bounds (above 100 m), −1 per step;
  episodes are cut off after 1000 steps (timeout)

## Setup

```
uv sync
```

Development:

```
uv run ruff check --fix .
uv run ruff format .
```

## Stage 1: Tabular Q-learning

Height and velocity are each split into 20 equal bins, giving a Q-table of shape `(20, 20, 3)`.
Training uses ε-greedy exploration (ε decays linearly from 1.0 to 0.05 over the first 80% of episodes),
α = 0.1, γ = 0.99.

```
uv run train-q                  # train 20k episodes, save runs/q_table.npy, evaluate
uv run train-q --episodes 5000  # shorter run
uv run train-q --watch          # load the Q-table and render one episode
```

### Results

Evaluation after 20k training episodes (1000 episodes, greedy policy, unseen seed):

| landed | crashed | out of bounds | timeout | mean impact velocity |
|---|---|---|---|---|
| 99.2% | 0.0% | 0.0% | 0.8% | 0.82 m/s |

<details>
<summary>Training log</summary>

```
episode   1000  eps 0.941  avg reward   -252.9  landed   0.0%  crashed   0.2%  out_of_bounds  99.8%  timeout   0.0%
episode   2000  eps 0.881  avg reward   -263.7  landed   0.1%  crashed   1.1%  out_of_bounds  98.8%  timeout   0.0%
episode   3000  eps 0.822  avg reward   -279.0  landed   0.2%  crashed   4.8%  out_of_bounds  95.0%  timeout   0.0%
episode   4000  eps 0.763  avg reward   -296.9  landed   0.5%  crashed  11.3%  out_of_bounds  88.2%  timeout   0.0%
episode   5000  eps 0.703  avg reward   -329.9  landed   1.6%  crashed  20.8%  out_of_bounds  77.6%  timeout   0.0%
episode   6000  eps 0.644  avg reward   -363.6  landed   9.8%  crashed  36.6%  out_of_bounds  53.0%  timeout   0.6%
episode   7000  eps 0.584  avg reward   -370.1  landed  10.9%  crashed  56.2%  out_of_bounds  30.8%  timeout   2.1%
episode   8000  eps 0.525  avg reward   -379.3  landed  11.3%  crashed  47.3%  out_of_bounds  39.8%  timeout   1.6%
episode   9000  eps 0.466  avg reward   -390.4  landed   9.2%  crashed  40.0%  out_of_bounds  47.8%  timeout   3.0%
episode  10000  eps 0.406  avg reward   -454.2  landed  10.5%  crashed  35.7%  out_of_bounds  48.3%  timeout   5.5%
episode  11000  eps 0.347  avg reward   -423.1  landed  24.2%  crashed  52.3%  out_of_bounds  12.5%  timeout  11.0%
episode  12000  eps 0.288  avg reward   -430.3  landed  28.6%  crashed  48.8%  out_of_bounds   8.6%  timeout  14.0%
episode  13000  eps 0.228  avg reward   -356.1  landed  38.6%  crashed  46.8%  out_of_bounds   3.6%  timeout  11.0%
episode  14000  eps 0.169  avg reward   -353.2  landed  40.7%  crashed  45.0%  out_of_bounds   0.4%  timeout  13.9%
episode  15000  eps 0.109  avg reward   -320.3  landed  45.5%  crashed  40.4%  out_of_bounds   1.3%  timeout  12.8%
episode  16000  eps 0.050  avg reward   -213.8  landed  59.0%  crashed  34.4%  out_of_bounds   0.0%  timeout   6.6%
episode  17000  eps 0.050  avg reward   -315.3  landed  50.3%  crashed  34.2%  out_of_bounds   0.1%  timeout  15.4%
episode  18000  eps 0.050  avg reward   -515.4  landed  38.2%  crashed  12.5%  out_of_bounds  11.0%  timeout  38.3%
episode  19000  eps 0.050  avg reward   -270.0  landed  49.9%  crashed  37.7%  out_of_bounds   1.9%  timeout  10.5%
episode  20000  eps 0.050  avg reward   -177.2  landed  65.4%  crashed  29.4%  out_of_bounds   0.0%  timeout   5.2%
training took 262.4 s
evaluation (1000 episodes): landed  99.2%  crashed   0.0%  out_of_bounds   0.0%  timeout   0.8%
mean impact velocity: 0.82 m/s
```

</details>

## Stage 2: DQN

_TODO_

## Stage 3: Double DQN

_TODO_
