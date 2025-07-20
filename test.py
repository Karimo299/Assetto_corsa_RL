import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from ACEnv import ACEnv
import numpy as np

env = DummyVecEnv([lambda: ACEnv()])

model_path = "models/SAC_First_Train/2200000_steps/SAC.zip"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = SAC.load(model_path, env=env, training=False)
obs = env.reset()

for _ in range(10):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

env.close()