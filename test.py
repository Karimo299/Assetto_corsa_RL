import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ACEnv import ACEnv
import numpy as np

# Create a fresh environment
env = DummyVecEnv([lambda: ACEnv()])
model_dir = "models/SAC_First_Train_20250721_143806/920000_steps"
model_path = os.path.join(model_dir, "SAC.zip")
vec_normalize_path = os.path.join(model_dir, "vec_normalize_stats.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(vec_normalize_path):
    raise FileNotFoundError(f"VecNormalize file not found: {vec_normalize_path}")

# Load the saved normalization statistics
# env = VecNormalize.load(vec_normalize_path, env)
env.training = False
env.norm_obs = True  # Set to True if you normalized obs during training
env.norm_reward = False  # IMPORTANT: Do NOT normalize rewards during evaluation

# Load the model with the normalized environment
model = SAC.load(model_path, env=env)
# Ensure the model is in evaluation mode
model.policy.set_training_mode(False)

# Reset the environment to start evaluation
obs = env.reset()

print("Starting evaluation runs...")
for episode in range(10):
    episode_reward = 0
    done = False
    step_count = 0
    
    while not done:
        # Get deterministic actions for evaluation
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, done, info = env.step(action)
        # Track metrics
        episode_reward += reward.item() if hasattr(reward, 'item') else reward
        step_count += 1
        # Optionally, print raw reward for debugging
        # print(f"Step reward (possibly normalized): {reward}")

    print(f"Episode {episode+1}: Steps: {step_count}, Reward: {episode_reward}")

env.close()
print("Evaluation complete")