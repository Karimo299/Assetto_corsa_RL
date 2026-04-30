import os
import torch
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from ACEnv import ACEnv


# --- Resume from checkpoint (set to a folder like "models/PPO_.../100000_steps") ---
RESUME_DIR = None

# --- Run name ---
RUN_NAME = "PPO_run"

# --- Training ---
TOTAL_TIMESTEPS = 10_000_000
SAVE_FREQ = 10_000

# --- PPO hyperparameters (only used for fresh runs) ---
LEARNING_RATE = 3e-4
N_STEPS = 4_096
BATCH_SIZE = 512
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = "cuda"

# --- Network architecture (only used for fresh runs) ---
POLICY_KWARGS = dict(
    net_arch=[256, 256, 128],
    activation_fn=torch.nn.ReLU,
)

# --- VecNormalize ---
NORM_OBS = True
NORM_REWARD = True
CLIP_OBS = 10.0


def make_env(log_path):
    return DummyVecEnv([lambda: Monitor(ACEnv(), log_path)])


def load_checkpoint(resume_dir, env):
    """
    Load PPO model and VecNormalize stats from a checkpoint folder.
    Returns (model, vec_env).
    """
    model_path = os.path.join(resume_dir, "PPO.zip")
    vec_norm_path = os.path.join(resume_dir, "vec_normalize_stats.pkl")

    # Load VecNormalize
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, env)
        print(f"VecNormalize loaded from {vec_norm_path}")
    else:
        print("No VecNormalize stats found, creating fresh wrapper.")
        vec_env = VecNormalize(
            env,
            norm_obs=NORM_OBS,
            norm_reward=NORM_REWARD,
            clip_obs=CLIP_OBS,
        )

    vec_env.training = True
    vec_env.norm_reward = NORM_REWARD
    vec_env.norm_obs = NORM_OBS

    # Load model
    model = PPO.load(
        model_path,
        env=vec_env,
        device=DEVICE,
    )
    print(f"Model loaded from {model_path}")

    return model, vec_env


def save_checkpoint(model, vec_env, save_dir, label=""):
    """Save PPO model and VecNormalize stats to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    tag = f"_{label}" if label else ""

    model_path = os.path.join(save_dir, f"PPO{tag}.zip")
    vec_norm_path = os.path.join(save_dir, f"vec_normalize_stats{tag}.pkl")

    model.save(model_path)
    vec_env.save(vec_norm_path)
    print(f"Checkpoint saved to {save_dir}")


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, vec_env, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            step_dir = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
            save_checkpoint(self.model, self.vec_env, step_dir)
        return True


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{RUN_NAME}_{timestamp}"
    log_path = f"./logs/{run_name}"
    models_path = f"./models/{run_name}"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    env = make_env(log_path)
    vec_env = None
    model = None

    try:
        if RESUME_DIR and os.path.exists(RESUME_DIR):
            print(f"Resuming from {RESUME_DIR}")
            model, vec_env = load_checkpoint(RESUME_DIR, env)
        else:
            print("Starting fresh run.")
            vec_env = VecNormalize(
                env,
                norm_obs=NORM_OBS,
                norm_reward=NORM_REWARD,
                clip_obs=CLIP_OBS,
            )

            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=LEARNING_RATE,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE,
                ent_coef=ENT_COEF,
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                policy_kwargs=POLICY_KWARGS,
                tensorboard_log=log_path,
                verbose=1,
                device=DEVICE,
            )

        callback = CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=models_path,
            vec_env=vec_env,
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name=RUN_NAME,
            callback=callback,
            reset_num_timesteps=RESUME_DIR is None,  # keep step count when resuming
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    finally:
        if model is not None and vec_env is not None:
            save_checkpoint(model, vec_env, models_path, label="final")
            print("Final checkpoint saved. Safe to exit.")
