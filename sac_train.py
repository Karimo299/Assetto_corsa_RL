import os
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from ACEnv import ACEnv


RESUME_DIR = None

# --- Run name ---
RUN_NAME = "SAC"

# --- Training ---
TOTAL_TIMESTEPS     = 10_000_000
SAVE_FREQ           = 10_000

# --- SAC hyperparameters (only used for fresh runs) ---
LEARNING_RATE       = 3e-4
BUFFER_SIZE         = 1_000_000
BATCH_SIZE          = 256
TAU                 = 0.005
GAMMA               = 0.99
TRAIN_FREQ          = 1
GRADIENT_STEPS      = 1
ENT_COEF            = "auto"
DEVICE              = "cuda"

# --- Network architecture (only used for fresh runs) ---
POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256, 128],
        qf=[256, 256, 128]
    ),
    activation_fn=torch.nn.ReLU
)

# --- VecNormalize ---
NORM_OBS            = True
NORM_REWARD         = True
CLIP_OBS            = 10.0

# =============================================================================


def make_env(log_path):
    return DummyVecEnv([lambda: Monitor(ACEnv(), log_path)])


def load_checkpoint(resume_dir, env):
    """
    Load model, VecNormalize stats and replay buffer from a checkpoint folder.
    Returns (model, vec_env).
    """
    model_path      = os.path.join(resume_dir, "SAC.zip")
    vec_norm_path   = os.path.join(resume_dir, "vec_normalize_stats.pkl")
    replay_buf_path = os.path.join(resume_dir, "replay_buffer.pkl")

    # Load VecNormalize
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, env)
        print(f"VecNormalize loaded from {vec_norm_path}")
    else:
        print("No VecNormalize stats found, creating fresh wrapper.")
        vec_env = VecNormalize(env, norm_obs=NORM_OBS, norm_reward=NORM_REWARD, clip_obs=CLIP_OBS)

    vec_env.training   = True
    vec_env.norm_reward = NORM_REWARD
    vec_env.norm_obs    = NORM_OBS

    # Load model
    model = SAC.load(model_path, env=vec_env, device=DEVICE)
    print(f"Model loaded from {model_path}")

    # Load replay buffer
    if os.path.exists(replay_buf_path):
        model.load_replay_buffer(replay_buf_path)
        print(f"Replay buffer loaded from {replay_buf_path}")
    else:
        print("No replay buffer found, starting with empty buffer.")

    return model, vec_env


def save_checkpoint(model, vec_env, save_dir, label=""):
    """Save model, VecNormalize stats and replay buffer to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    tag = f"_{label}" if label else ""

    model_path      = os.path.join(save_dir, f"SAC{tag}.zip")
    vec_norm_path   = os.path.join(save_dir, f"vec_normalize_stats{tag}.pkl")
    replay_buf_path = os.path.join(save_dir, f"replay_buffer{tag}.pkl")

    model.save(model_path)
    vec_env.save(vec_norm_path)
    model.save_replay_buffer(replay_buf_path)
    print(f"Checkpoint saved to {save_dir}")


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, vec_env, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_env   = vec_env

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            step_dir = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
            save_checkpoint(self.model, self.vec_env, step_dir)
        return True


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name    = f"{RUN_NAME}_{timestamp}"
    log_path    = f"./logs/{run_name}"
    models_path = f"./models/{run_name}"
    os.makedirs(log_path,    exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    env     = make_env(log_path)
    vec_env = None
    model   = None

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
                clip_obs=CLIP_OBS
            )
            n_actions    = env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.2 * np.ones(n_actions)
            )
            model = SAC(
                "MlpPolicy",
                vec_env,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                tau=TAU,
                gamma=GAMMA,
                train_freq=TRAIN_FREQ,
                gradient_steps=GRADIENT_STEPS,
                ent_coef=ENT_COEF,
                policy_kwargs=POLICY_KWARGS,
                verbose=1,
                tensorboard_log=log_path,
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