# Performance-Oriented Autonomous Racing Under Minimal Perception Using Pure RL

A SAC-based reinforcement learning agent trained to race on the Red Bull Ring in Assetto Corsa
using only 12 telemetry values + 42 ray sensors, no privileged state, no trajectory guidance, no expert demos.

**CS4490Z Undergraduate Thesis - Western University**  
Karim Abousamra | Supervisor: Umair Rehman

---

## Demo

| Stage | Description | Video |
|-------|-------------|-------|
| Stage 1 | Initial policy erratic steering, avg ~1:25 | [![Stage 1](https://img.youtube.com/vi/tZgyMhIlDg0/0.jpg)](https://youtu.be/tZgyMhIlDg0) |
| Stage 2 | Reward rebalanced toward smoothness | [![Stage 2](https://img.youtube.com/vi/jSJssUqp0Oo/0.jpg)](https://youtu.be/jSJssUqp0Oo) |
| Stage 3 | Enhanced policy with more speed emphasis, avg ~1:17 | [![Stage 3](https://img.youtube.com/vi/2MjjqL9aYKQ/0.jpg)](https://youtu.be/2MjjqL9aYKQ) |

---

## Environment

- **Simulator:** Assetto Corsa (CSP shared memory interface, 20 Hz)
- **Track:** Red Bull Ring
- **Observation space:** 12 telemetry values (normalized car position, speed, steer angle, last steering, last throttle/brake, lateral velocity, longitudinal velocity, yaw rate, average tyre slip, gas, brake, DRS available) + 42 ray-based distance sensors
- **Action space:** Continuous 2D - steering + unified throttle/brake signal
- **Wrapped as:** OpenAI Gym-compatible environment

## Algorithm & Hyperparameters

SAC was selected after evaluating PPO (slow convergence) and TD3 (suboptimal, no lap completion).

| Parameter | Value |
|-----------|-------|
| Architecture | MLP [256, 256, 128] + ReLU |
| Learning rate | 3 × 10⁻⁴ |
| Replay buffer | 1,000,000 |
| Batch size | 256 |
| γ / τ | 0.99 / 0.005 |
| Entropy tuning | Automatic |

---

## Project Structure

```
assetto_rl/
  env/            # ACEnv (Gym env), sim_info, track_utils, car_control
  train/          # sac_train.py, ppo_train.py, td3_train.py
  eval/           # test.py (evaluate a policy), test_env.py (env sanity check)
tools/            # render_track.py (track/ray visualizer)
csv/              # track + racing-line data
models/           # trained checkpoints (not in git - see Testing)
```

## Setup

```bash
git clone https://github.com/Karimo299/Assetto_corsa_RL.git
cd Assetto_corsa_RL
pip install -r requirements.txt
```

Requires Assetto Corsa with Content Manager + CSP installed, with the CSP shared-memory
interface enabled so the env can read telemetry and send controls.

> All scripts are modules under the `assetto_rl` package, so run them with `python -m ...`
> **from the repository root** (not `python path/to/file.py`).

## Training

Training uses SAC + VecNormalize with periodic checkpoints ([sac_train.py](assetto_rl/train/sac_train.py)).

1. Configure the run at the top of `assetto_rl/train/sac_train.py`:
   - `RUN_NAME`, `TOTAL_TIMESTEPS`, `SAVE_FREQ`
   - `RESUME_DIR` - a checkpoint folder to resume from, or `None` for a fresh run
2. Start training:

```bash
python -m assetto_rl.train.sac_train
```

Checkpoints are saved under `models/<RUN_NAME>_<timestamp>/<steps>_steps/`, each containing
`SAC.zip`, `vec_normalize_stats.pkl`, and `replay_buffer.pkl`. PPO and TD3 baselines live
alongside as `ppo_train.py` / `td3_train.py`.

## Testing

Evaluate a trained policy with [test.py](assetto_rl/eval/test.py).

1. Download the trained models from the [Releases](https://github.com/Karimo299/Assetto_corsa_RL/releases)
   page and unzip them into the project root so `models/...` exists.
2. In `assetto_rl/eval/test.py`, uncomment the `model_dir` you want (leave only one active).
3. Run:

```bash
python -m assetto_rl.eval.test
```

The script runs 10 deterministic episodes and prints per-episode steps and reward.
To verify the environment alone (no model), run `python -m assetto_rl.eval.test_env`.

## Track Visualizer

[render_track.py](tools/render_track.py) is a live pygame debug view that reads telemetry
over the CSP shared-memory interface (Assetto Corsa must be running and on track). It draws
the track barriers, the car, and all 42 rays, with an overlay for position, heading, speed,
throttle/brake/steer, on-track status, lap progress, and straight/turn detection. Scroll to
zoom. It also prints the current spawn **teleport tuple** to the console for use as a start
point in `ACEnv`.

```bash
python -m tools.render_track
```

---

## Results

| Driver | Best Lap | Mean Speed | Mean Throttle | Avg Slip |
|--------|----------|------------|---------------|----------|
| **Agent** | 1:17.094 | 199.4 km/h | 0.792 | 0.403 |
| Built-in AI | 1:11.875 | 214.2 km/h | 0.788 | 0.278 |
| Karim | 1:10.674 | 217.0 km/h | 0.798 | 0.285 |
| Saaim | 1:09.670 | 220.4 km/h | 0.749 | 0.320 |
| Ziad | 1:12.859 | 210.2 km/h | 0.746 | 0.320 |

---

## Limitations
- Single-track only (Red Bull Ring)
- Single simulator instance - no parallel training
- Simulation only, no sim-to-real evaluation
- ~5 sec gap to built-in AI remains