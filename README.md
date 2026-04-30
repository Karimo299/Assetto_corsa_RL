# Performance-Oriented Autonomous Racing Under Minimal Perception Using Pure RL

A SAC-based reinforcement learning agent trained to race on the Red Bull Ring in Assetto Corsa
using only 12 telemetry values + 42 ray sensors, no privileged state, no trajectory guidance, no expert demos.

**CS4490Z Undergraduate Thesis — Western University**  
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
- **Observation space:** 12 telemetry values (speed, steering, yaw rate, tyre slip, etc.) + 42 ray-based distance sensors
- **Action space:** Continuous 2D — steering + unified throttle/brake signal
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

## Setup

```bash
git clone https://github.com/Karimo299/Assetto_corsa_RL.git
cd Assetto_corsa_RL
pip install -r requirements.txt
```

Requires Assetto Corsa with Content Manager + CSP installed. See `docs/environment_setup.md` for shared memory configuration.

## Training

Training is run with `sac_train.py` (SAC + VecNormalize + periodic checkpoints).

1. Configure run settings in `sac_train.py`:
   - `RUN_NAME`, `TOTAL_TIMESTEPS`, `SAVE_FREQ`
   - `RESUME_DIR` (set to existing checkpoint folder to resume, or disable for a fresh run)
2. Start training:

```bash
python sac_train.py
```

Checkpoints are saved under `models/<RUN_NAME>_<timestamp>/` with step-based subfolders containing:
- `SAC.zip`
- `vec_normalize_stats.pkl`
- `replay_buffer.pkl`

## Testing

Evaluate a trained policy with `test.py`.

1. Unzip `models.zip` into the project root so `models/...` exists.
2. Open `test.py` and uncomment the `model_dir` you want to test (leave only one active `model_dir` line).
3. Run evaluation:

```bash
python test.py
```
The script runs 10 deterministic episodes and prints per-episode steps and reward.

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
- Single simulator instance — no parallel training
- Simulation only, no sim-to-real evaluation
- ~5 sec gap to built-in AI remains