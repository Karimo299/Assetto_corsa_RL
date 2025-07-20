import os
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from ACEnv import ACEnv  # Replace with your actual F1 environment import
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch
from datetime import datetime


class SaveVecNormalizeCallback(BaseCallback):
  def __init__(self, save_freq, save_path, verbose=1, env=None):
    super(SaveVecNormalizeCallback, self).__init__(verbose)
    self.save_freq = save_freq
    self.save_path = save_path
    self.env = env

  # def _on_rollout_start(self):
  #   self.env.env_method("resume_game")

  # def _on_rollout_end(self):
  #   self.env.env_method("pause_game")

  def _on_step(self) -> bool:
    if self.num_timesteps % self.save_freq == 0:

      save_path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
      os.makedirs(save_path, exist_ok=True)

      # Save the model
      model_path = os.path.join(save_path, f"{self.model.__class__.__name__}.zip")
      self.model.save(model_path)
      if self.verbose > 0:
        print(f"Model saved at step {self.num_timesteps} to {model_path}")

      # Save VecNormalize stats (always use the env passed to the callback, not self.training_env)
      if self.env is not None and hasattr(self.env, 'save'):
        vec_norm_path = os.path.join(save_path, f"vec_normalize_stats.pkl")
        self.env.save(vec_norm_path)
        if self.verbose > 0:
          print(f"VecNormalize stats saved to {vec_norm_path}")
       # Save the replay buffer

      replay_buffer_path = os.path.join(save_path, "replay_buffer.pkl")
      self.model.save_replay_buffer(replay_buffer_path)
      if self.verbose > 0:
        print(f"Replay buffer saved to {replay_buffer_path}")

    # Print normalized rewards
    if isinstance(self.training_env, VecNormalize):
      rewards = self.locals["rewards"]  # Raw rewards from the environment
      normalized_rewards = self.training_env.normalize_reward(rewards)
      unnormalize_rewards = self.training_env.unnormalize_reward(rewards)
      # print(f"Raw rewards: {unnormalize_rewards}, Normalized rewards: {normalized_rewards}")

    return True


pretrained_vec_normalize_path = None
pretrained_model_path = None
replay_buffer_path = None
# pretrained_model_path = "models/SAC_First_Train/2200000_steps/SAC.zip"
# pretrained_model_path = "models/SAC_First_Train/End_model.zip"
# pretrained_vec_normalize_path = "models/SAC_First_Train/vec_normalize_stats.pkl"
# # replay_buffer_path = "./models/plswork/replay_buffer_end.pkl"


# Set paths
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
name = f"SAC_First_Train_{timestamp}"
log_path = f"./logs/{name}"
models_path = f"./models/{name}"
os.makedirs(log_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)


vec_env = None
model = None

if __name__ == "__main__":
  try:
    # Create the base environment and wrap it in VecNormalize
    env = DummyVecEnv([lambda: Monitor(ACEnv(), log_path)])  # Wrap with Monitor
    vec_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if replay_buffer_path and os.path.exists(replay_buffer_path):
      model.load_replay_buffer(replay_buffer_path)
      print("Replay buffer loaded.")

    # Load VecNormalize stats if available
    if pretrained_vec_normalize_path and os.path.exists(pretrained_vec_normalize_path):
      vec_env = VecNormalize.load(pretrained_vec_normalize_path, env)  # Automatically wraps env
      print(f"VecNormalize stats loaded from {pretrained_vec_normalize_path}")
    else:
      print(f"No VecNormalize stats found at {pretrained_vec_normalize_path}. Using new VecNormalize wrapper.")

    vec_env.training = True  # Ensure training mode is enabled
    vec_env.norm_reward = True  # Normalize rewards
    vec_env.norm_obs = True  # Normalize observations

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Actor (policy) network
            qf=[256, 256, 128]   # Critic (Q-value) network
        ),
        activation_fn=torch.nn.ReLU  # Activation function
    )

    # Load the pre-trained SAC model
    if pretrained_model_path and os.path.exists(pretrained_model_path):
      model = SAC.load(pretrained_model_path, env=vec_env, tensorboard_log=log_path, verbose=2)
      print(f"Pre-trained SAC model loaded from {pretrained_model_path}")
    else:
      # policy_kwargs = {"net_arch": [400, 300, 128]}  # Example architecture, modify as needed
      n_actions = env.action_space.shape[0]
      action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

      model = SAC(
          "MlpPolicy",
          env,
          learning_rate=3e-4,
          buffer_size=1_000_000,
          batch_size=256,
          tau=0.005,
          gamma=0.99,
          train_freq=1,
          gradient_steps=1,
          ent_coef="auto",
          policy_kwargs=policy_kwargs,
          verbose=1,
          tensorboard_log=log_path,
      )

    # Optional: Continue training the pre-trained model
    save_callback = SaveVecNormalizeCallback(save_freq=10000, save_path=models_path, env=vec_env)
    total_timesteps = 10_000_000  # Adjust as needed for additional training
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="SAC",
        callback=save_callback,
        reset_num_timesteps=True
    )

  except KeyboardInterrupt:

    model.save(f"{models_path}/End_model.zip")
    vec_env.save(f"{models_path}/vec_normalize_stats_end.pkl")

    print(model.seed)
    if hasattr(model, 'save_replay_buffer'):
      replay_buffer_end_path = os.path.join(models_path, "replay_buffer_end.pkl")
      model.save_replay_buffer(replay_buffer_end_path)
      print(f"Replay buffer saved to {replay_buffer_end_path}")
