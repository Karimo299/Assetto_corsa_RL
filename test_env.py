import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from ACEnv import ACEnv

# Import your custom environment here
# If your environment is in a different file, import it like:
# from your_module import YourEnv

# For testing purposes, let's assume you have a custom environment class
# Replace this with your actual environment import
try:
    print("Checking custom environment...")
    
    # Create an instance of your environment
    env = ACEnv()
    
    # Check if the environment follows the Gym API
    check_env(env)
    
    print("Environment check passed!")
    
    # Optionally, you can test some basic interactions with the environment
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    action = env.action_space.sample()
    print(f"Random action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    print(f"Observation shape: {obs.shape}")
    print(f"Info: {info}")
    
    env.close()

except ImportError as e:
    print(f"Error importing environment: {e}")
    print("Make sure your environment class is properly defined and imported.")
except Exception as e:
    print(f"Error during environment checking: {e}")