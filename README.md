# Self-Driving Car with Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyGame](https://img.shields.io/badge/PyGame-2.5.0-green.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An interactive reinforcement learning project where an AI agent learns to drive a car on a custom track using PPO (Proximal Policy Optimization). The focus of this project is not just making it work — but understanding *why* it works through systematic reward shaping and hyperparameter tuning.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Policy Optimization — What I Actually Learned](#policy-optimization--what-i-actually-learned)
- [Reward Function Journey — Every Iteration](#reward-function-journey--every-iteration)
- [Installation](#installation)
- [Usage](#usage)
- [Fine-Tuning Guide](#fine-tuning-guide)
- [Monitoring Training](#monitoring-training)
- [Saving and Loading Models](#saving-and-loading-models)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Features

- PPO algorithm from Stable-Baselines3 with carefully tuned hyperparameters
- 5 radar-style distance sensors using ray projection
- Start/Stop buttons for controlled, interruptible training sessions
- Real-time visualization with live stats (episode, steps, distance, best)
- Automatic model saving on training completion or interruption
- TensorBoard integration for tracking policy loss, value loss, and reward curves
- Modifiable track layout and reward function

---

## Architecture

```
+-----------------------+
|   5 Radar Sensors     |  Measures normalized distance to walls (ray projection)
+-----------+-----------+
            |
            v
+-----------------------+
|   Neural Network      |  PPO Policy — 2-layer MLP
+-----------+-----------+
            |
            v
+-----------------------+
|   3 Discrete Actions  |  Turn Left / Go Straight / Turn Right
+-----------------------+
```

### Sensors

The car has 5 distance sensors positioned at:

| Sensor | Angle | Direction |
|--------|-------|-----------|
| 1 | -90° | Left |
| 2 | -45° | Front-Left |
| 3 | 0° | Front |
| 4 | +45° | Front-Right |
| 5 | +90° | Right |

Each sensor returns a value normalized between 0 and 1 (0 = wall contact, 1 = maximum range).

### Action Space

```python
0 = Turn Left   (7° rotation)
1 = Go Straight (no rotation)
2 = Turn Right  (7° rotation)
```

---

## How It Works

### Training Loop

Each episode starts with the car at a fixed spawn point. At every timestep, the 5 sensor values are fed into the neural network, which outputs a probability distribution over the 3 actions. PPO samples an action, applies it, computes the reward, and stores the experience. After `n_steps = 2048` timesteps, the collected batch is used to update the policy.

### PPO Hyperparameters

```python
learning_rate = 1e-4      # Chosen after testing 4 values — see below
n_steps = 2048            # Steps before policy update
batch_size = 64           # Training batch size
n_epochs = 10             # Optimization passes per update
gamma = 0.99              # Discount factor
gae_lambda = 0.95         # Advantage estimation smoothing
clip_range = 0.2          # PPO clipping parameter (epsilon)
ent_coef = 0.01           # Entropy bonus — encourages exploration
```

---

## Policy Optimization — What I Actually Learned

### Why Vanilla Policy Gradient Fails Here

In basic policy gradient, the update rule is:

```
gradient = log_prob(action) * reward
```

This is unstable. If a single lucky episode gives a large reward, the policy can update too aggressively — destroying previously learned behavior. In driving, this manifests as the car suddenly forgetting how to corner after a long straight run gives outsized returns.

### How PPO Fixes This

PPO constrains each policy update using a clipped objective:

```
L_CLIP = min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t)
```

Where:
- `r_t` = ratio of new policy probability to old policy probability
- `A_t` = advantage estimate — how much better this action was compared to the baseline
- `ε` = clip range (0.2 in this project)

If the policy tries to change too aggressively (ratio moves far from 1.0), the clip cuts off the gradient. The policy is constrained to stay close to the old policy — learning in small, stable steps rather than large, destructive ones. This is what makes PPO reliable for continuous control problems like driving.

### Learning Rate Selection — Why 1e-4

I tested four values systematically:

| Learning Rate | Result |
|---------------|--------|
| `1e-3` | Policy updates too large even with clipping. Agent unlearned good behavior between episodes. Training was noisy and diverged on complex corners. |
| `3e-4` | Standard recommendation from Stable-Baselines3 docs. Works for many tasks, but too aggressive for smooth driving — steering became jerky. Policy oscillated rather than converging. |
| `1e-4` | Sweet spot. Updates small enough that the policy evolved smoothly. Training is slower but stable. Cornering improved monotonically across episodes. |
| `3e-5` | Effectively stopped learning. Gradient signal too small to overcome initialization noise. |

The takeaway: learning rate interacts with the task structure. For driving — where smooth, incremental policy improvement matters more than fast convergence — `1e-4` consistently outperformed the SB3 default.

---

## Reward Function Journey — Every Iteration

This section documents each iteration of the reward function honestly — what I tried, what broke, and what actually worked. The final function is the result of four complete rewrites.

### Iteration 1 — Survival Reward

```python
reward = 1.0       # every timestep alive
if collision:
    reward = -100.0
```

**Result:** The agent learned to drive in slow circles in open areas of the track. It never crashed. It also never went anywhere useful.

**What broke:** This is a textbook case of reward hacking — the agent found a policy that maximized survival without achieving the intended behavior. There was no incentive to make progress, so it did not.

**Lesson:** Rewards must incentivize the behavior you want, not just the final outcome.

---

### Iteration 2 — Forward Velocity

```python
reward = velocity_forward * 1.0
if collision:
    reward = -10.0
```

**Result:** The agent drove at maximum speed directly into walls. The cumulative velocity reward over hundreds of timesteps dwarfed the single-timestep collision penalty.

**What broke:** The ratio between cumulative reward and terminal penalty was completely off. The agent correctly computed that high-speed driving was worth more in expectation than collision avoidance.

**Lesson:** Reward magnitudes matter as much as reward structure. The relative scale of incentives and penalties determines the policy, not their signs.

---

### Iteration 3 — Angular Velocity Penalty

```python
reward = velocity_forward * 0.5
reward -= abs(angular_velocity) * 0.3   # penalize spinning
if collision:
    reward = -50.0
```

**Result:** Noticeably better. The agent moved forward and stopped spinning. But it learned to drive along walls — grazing them without triggering the binary collision threshold. No concept of "danger zone" existed.

**What broke:** A binary collision penalty has no gradient. The agent received zero signal about proximity to walls until the exact moment of contact. There was nothing to tell it that 5 pixels from a wall is dangerous.

**Lesson:** Binary penalties teach nothing about proximity. You need a continuous signal — a gradient of discomfort as walls approach.

---

### Iteration 4 — The One That Worked

```python
# Incentivize smooth forward progress
reward = velocity_forward * 0.3

# Penalize erratic steering
reward -= abs(angular_velocity) * 0.2

# Soft wall proximity penalty — grows continuously as walls approach
min_ray = min(ray_distances)        # closest wall across all 5 sensors
if min_ray < 0.3:                   # within 30% of max sensor range
    reward -= (0.3 - min_ray) * 2.0 # penalty magnitude grows as gap shrinks

# Hard terminal penalty on collision
if collision:
    reward = -5.0
    done = True

# Lap completion bonus
if lap_complete:
    reward += 10.0
```

**What changed:** The soft proximity penalty creates a continuous signal before collision occurs. As the car drifts toward a wall, the reward begins decreasing — the agent learns discomfort *before* contact, not only at the moment of contact.

**Why this is reward shaping:** Adding intermediate signals that guide the agent toward the final goal without changing the optimal policy is formally called reward shaping. The key requirement is that the shaped reward must not create new optimal policies that differ from the original — proximity discomfort satisfies this because avoiding walls and completing laps remain aligned.

**Why the smaller collision penalty (-5 vs -100) worked better:** With a smaller terminal penalty, the agent was not paralyzed by collision fear. It still explored, but was guided away from walls continuously rather than scared away discontinuously.

### Final Reward Structure Summary

| Signal | Type | Magnitude | Purpose |
|--------|------|-----------|---------|
| Forward velocity | Continuous positive | 0.3x | Incentivize progress |
| Angular velocity | Continuous negative | -0.2x | Penalize erratic steering |
| Wall proximity | Continuous negative, graduated | up to -0.6 | Create danger gradient |
| Collision | Terminal negative | -5.0 | Hard boundary |
| Lap complete | Sparse positive | +10.0 | Long-horizon goal |

---

## Training Progress

| Episode Range | Behavior |
|---------------|----------|
| 1–50 | Crashes immediately, no wall awareness |
| 50–200 | Learns basic wall avoidance, erratic steering |
| 200–400 | Develops corner awareness, smoother trajectories |
| 400–500+ | Consistent lap completion, policy stable |

### Live Stats Display

```
Episode: 523
Steps:   45,231
Distance: 2,847
Best:    3,102
Status:  TRAINING
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip
- 2GB RAM minimum
- Display output (X11 or equivalent for visualization)

### 1. Clone the Repository

```bash
git clone https://github.com/Soorya2201/RL_Training-of-vehicle-agent-using-Stable_Baseline3
```

### 2. Install Dependencies

```bash
pip install pygame numpy gymnasium stable-baselines3
```

Or from requirements file:

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python rl_car.py
```

---

## Usage

### Starting Training

1. Launch the program:
   ```bash
   python main2.py
   ```
2. Click the **START** button (green) to begin training.
3. Click **STOP** (red) to pause at any time.
4. Resume by clicking START — training continues from the last checkpoint.

### Keyboard Shortcuts

- `ESC` or close window: exit
- `Ctrl+C` in terminal: interrupt and auto-save model

---

## Fine-Tuning Guide

### Learning Rate

```python
learning_rate = 1e-3   # Faster but unstable (not recommended for driving)
learning_rate = 3e-4   # SB3 default — good for most tasks
learning_rate = 1e-4   # Best for smooth driving (used in this project)
learning_rate = 3e-5   # Too slow — effectively stops learning
```

### Reward Function

In the `step()` method:

```python
# Increase forward progress weight
reward += velocity_forward * 0.5         # was 0.3

# Tighten wall proximity threshold
if min_ray < 0.4:                        # catch danger earlier
    reward -= (0.4 - min_ray) * 2.0

# Add a speed bonus
reward += car.speed * 0.1
```

### Track Layout

Edit `draw_track()` in the `TrackEnv` class:

```python
def draw_track(self):
    points = [
        (x1, y1), (x2, y2), (x3, y3), ...
    ]
    pygame.draw.lines(self.map_surface, ROAD_COLOR, True, points, 120)
    #                                                              ^^^ track width
```

### Training Duration

```python
model.learn(total_timesteps=500_000)   # default is 100,000
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=./ppo_car_tensorboard/
# Open: http://localhost:6006
```

Metrics tracked: episode reward, episode length, policy loss, value loss, learning rate.

### Console Output

```
Episode finished — Reward: 245.32,  Length: 387
Episode finished — Reward: 512.18,  Length: 891
Episode finished — Reward: 1023.45, Length: 1456
```

---

## Saving and Loading Models

Models auto-save on training completion or `Ctrl+C` interrupt.

### Manual Save

```python
model.save("my_trained_car")
```

### Loading a Trained Model

```python
from stable_baselines3 import PPO

model = PPO.load("ppo_self_driving_car")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()
```

---

## Configuration Reference

### Display

```python
WIDTH, HEIGHT = 1200, 800   # Window resolution
FPS = 60                    # Frame rate (reduce to speed up training)
SENSOR_LENGTH = 200         # Ray projection range
```

### Car

```python
CAR_SIZE = 20               # Dimensions in pixels
speed = 5                   # Movement speed per frame
angle = 0                   # Starting orientation
```

### Colors

```python
ROAD_COLOR   = (100, 100, 100)  # Gray
CAR_COLOR    = (255, 0,   0  )  # Red
SENSOR_COLOR = (0,   255, 0  )  # Green
```

---

## Troubleshooting

**Car crashes immediately every episode**
Increase `SENSOR_LENGTH` to 250–300, reduce `speed` to 3–4, or widen the track.

**Training is too slow**
Reduce FPS to 30 for faster simulation, increase `learning_rate` to `3e-4`, or reduce `n_steps` to 1024.

**Agent learned good behavior then forgot it**
Decrease `learning_rate` to `1e-4`, increase `batch_size` to 128, and check the reward function for conflicting signals (e.g., angular penalty fighting with the lap bonus).

**`pygame not found` error**

```bash
pip install pygame --upgrade
```

---

## Roadmap

- Multiple track layouts with increasing difficulty (curriculum learning)
- Manual control mode for human baseline comparison
- Track editor UI
- Dynamic obstacles
- Continuous action space support via SAC or TD3
- Multi-agent racing

---

## Learning Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm — Original Paper](https://arxiv.org/abs/1707.06347)
- [Deep RL Course by Hugging Face](https://huggingface.co/deep-rl-course)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push and open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Stable-Baselines3 team for the RL library
- OpenAI for PPO algorithm research
- Pygame community for the graphics framework

---

Made by Soorya Narayanan
