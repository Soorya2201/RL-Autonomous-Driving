# Self-Driving Car with Reinforcement Learning

**A PPO agent that learns to drive a closed track using only five ray-cast distance sensors, in a custom Gymnasium environment built on PyGame. This README documents how reward design and learning rate changed what the agent learned, using the training logs committed in this repo.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-custom%20env-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**[Watch the trained agent drive](https://drive.google.com/file/d/1moOJdhn-2XTUbPm8Y2I-_lR_76FmFDR2/view)**

## Highlights

- **Custom Gymnasium environment** (`main2.py`): a 1200×800 track, pixel-level collision detection, 5 ray-cast sensors at -90°, -45°, 0°, 45° and 90°, and 3 discrete steering actions.
- **18x longer survival.** Over 100k PPO timesteps, the mean episode length rose from **28 to 510 steps** before crashing (TensorBoard run `PPO_4`).
- **A controlled learning-rate comparison** across 1e-3, 3e-4 and 1e-4, read from the committed TensorBoard logs.
- **Reward hacking found and fixed.** The first reward (`main.py`) taught the agent to survive without making progress. The shaped reward in `main2.py` fixes that by rewarding distance, penalizing steering, and rewarding clearance from walls.
- **Interactive training.** Start and Stop buttons, live rendering, TensorBoard logging, and automatic saving on completion or interrupt.

## Environment

| | |
|---|---|
| Observation | 5 sensor distances, each normalized to [0, 1] (200 px max range) |
| Actions | 0 = turn left 7°, 1 = straight, 2 = turn right 7°; constant speed |
| Termination | The car's collision points touch off-road pixels |
| Policy | Stable-Baselines3 `MlpPolicy` (actor-critic MLP) |

```
5 ray-cast sensors ──► PPO MlpPolicy ──► {left, straight, right}
        ▲                                        │
        └──────── PyGame track, collision check ◄┘
```

## Reward design

**v1 — survival reward (`main.py`)**

```python
reward = 1          # every step alive
if crashed:
    reward = -100
```

The agent learned to stay alive without driving anywhere useful. This is textbook reward hacking: it was rewarded for surviving, not for progress.

**v2 — shaped reward (`main2.py`, the committed final version)**

```python
if crashed:
    reward = -100 + distance_traveled * 0.1      # a crash after real progress hurts less
else:
    reward = 2 + distance_traveled * 0.01        # alive + progress
    if action != STRAIGHT:
        reward -= 1                              # discourage zig-zagging
    if mean(sensor_distances) > 50:
        reward += 2                              # stay centered, away from walls
```

**What the iterations taught me**

- **Reward what you want, not a proxy for it.** Survival alone produced circling. Progress has to appear in the reward.
- **Relative magnitudes decide the policy.** If per-step progress reward dwarfs the crash penalty, the agent races into walls. Scaling the crash penalty by distance traveled kept both signals meaningful.
- **Binary penalties carry no gradient.** A crash-only penalty says nothing about being 5 px from a wall. A continuous clearance term gives the agent a signal before contact.
- **Shaping can change the optimum.** Only potential-based shaping (Ng et al., 1999) is guaranteed to preserve the optimal policy. The turn penalty and clearance bonus here are not potential-based, so they bias the agent toward smooth, centered driving. That is intended for this task, but it is a design choice, not a free lunch.

## Learning-rate comparison (from `ppo_car_tensorboard/`)

Mean episode length, in steps survived before crashing:

| Run | Learning rate | at 10k steps | at 22.5k steps | final |
|---|---|---:|---:|---:|
| `PPO_2` | 1e-3 | 82.8 | 174.3 | 220.2 at 26.6k |
| `PPO_4` | **3e-4** | 72.5 | 162.9 | **510.4 at 100k** |
| `PPO_3` | 1e-4 | 47.5 | 78.6 | 78.6 at 22.5k |

- **1e-4** learned roughly half as fast as the others over the same number of steps.
- **1e-3** learned fastest early, but that run was stopped at 26.6k steps, so its long-run stability is untested here.
- **3e-4** (the Stable-Baselines3 default) kept improving steadily: 296 steps at 51k, 443 at 76k, and 510 at 100k. It is the setting used in `main2.py`.

Caveat: the runs differ in length, and the 1e-3 run matches the `main.py` configuration (v1 reward), so these numbers compare learning dynamics, not final skill. Running every learning rate for 100k steps on the v2 reward across several seeds is the next step.

## PPO configuration (`main2.py`)

```python
PPO("MlpPolicy", env,
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    tensorboard_log="./ppo_car_tensorboard/")
model.learn(total_timesteps=100_000)
```

PPO's clipped objective, `min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)` with ε = 0.2, stops one lucky episode from dragging the policy too far in a single update.

## Run it

```bash
pip install pygame numpy gymnasium stable-baselines3 tensorboard
python main2.py                                # opens the window; click Start to train
tensorboard --logdir ppo_car_tensorboard       # learning curves
```

The model is saved to `ppo_self_driving_car.zip`, or to `ppo_self_driving_car_interrupted.zip` if training is stopped. A trained model is included:

```python
from stable_baselines3 import PPO
model = PPO.load("ppo_self_driving_car")
```

`main.py` is the original v1 environment, kept for comparison.

## Repository layout

```
main2.py                    final environment, shaped reward, training loop and UI
main.py                     v1 environment with the survival reward (reward-hacking baseline)
ppo_self_driving_car.zip    trained PPO policy
ppo_car_tensorboard/        TensorBoard logs for the runs above
```

## Roadmap

- Equal-length, multi-seed learning-rate sweeps on the v2 reward
- A potential-based clearance term, then compare against the current shaping
- Curriculum across several tracks, and generalization to an unseen track
- Continuous steering and throttle with SAC or TD3

## License

MIT — see [LICENSE](LICENSE).
