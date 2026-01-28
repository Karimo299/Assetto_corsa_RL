import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from CarControl import CarController
from sim_info import get_car_details, ray_angles
from track_utils import load_track_data, create_continuous_track_polygon, calculate_ray_endpoint, is_car_off_track


_, left_barrier, right_barrier = load_track_data()
left_polygon = create_continuous_track_polygon(left_barrier)
right_polygon = create_continuous_track_polygon(right_barrier)


class ACEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.car_controller = CarController()

        self.in_lap = False
        self.car_pos = [0, 0]
        
        self.total_reward = 0.0
        # NEEDED FOR DIRECT MEMORY ACCESS ASETTO CORSA
        self.INIT_TELEPORT_POS = (490, -11, 248, -0.85, 0, 0.24, 2)
        self.RESET_TELEPORT_POS = (0, 0, 0, 1, 0, 0, 0)
        self.INIT_CONTROL = (1, 0, 0)


        self.OBS_BASE_FEATURES = 4
        
        # Rate limiting: 20Hz = 1/20 seconds per step
        self.step_duration = 1.0 / 20.0

        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        ray_obs_low = np.zeros(len(ray_angles), dtype=np.float32)
        ray_obs_high = np.ones(len(ray_angles), dtype=np.float32) * 5000
        low = np.concatenate([np.array([0, 0, -1, -1], dtype=np.float32), ray_obs_low])
        high = np.concatenate([np.array([1, 350, 1, 1], dtype=np.float32), ray_obs_high])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.obs_size = self.OBS_BASE_FEATURES + len(ray_angles)

    def reset(self, seed=None, options=None):
        self.in_lap = False
        self.total_reward = 0.0


        # Teleport to initial position
        # ALL OF THESE ARE NEEDED DONT REMOVE ANY OF THEM
        self.car_controller.teleport(*self.INIT_TELEPORT_POS)
        self.car_controller.teleport(*self.RESET_TELEPORT_POS)
        self.car_controller.write_car_controls(*self.INIT_CONTROL)
        # END OF ALL OF THESE ARE NEEDED DONT REMOVE ANY OF THEM


        reset_state = np.zeros(self.obs_size, dtype=np.float32)
        return reset_state, {}

    def step(self, action):
        step_start = time.time()
        
        steering = float(action[0])
        throttle_brake = float(action[1])

        throttle = max(0, throttle_brake)
        brake = max(0, -throttle_brake)

        # Get car state
        car_pos, heading, speed, gas, brake, steerAngle, normalizedCarPosition, distance_traveled, laps, drs_available = get_car_details()
        self.car_pos = car_pos

        # Track lap progress
        if normalizedCarPosition < 0.1 and normalizedCarPosition >= 0 and not self.in_lap:
            self.in_lap = True
        if not self.in_lap:
            normalizedCarPosition = 0


        #TODO clean this up
        # Calculate ray distances
        ray_distances = []
        for angle in ray_angles:
            ray_endpoint = calculate_ray_endpoint(self.car_pos, heading, angle, left_polygon, right_polygon)
            if ray_endpoint is not None:
                distance = np.linalg.norm(np.array(self.car_pos) - np.array(ray_endpoint))
                ray_distances.append(distance)
            else:
                ray_distances.append(5000)
        ray_distances = np.array(ray_distances, dtype=np.float32)


        
        # Apply controls
        self.car_controller.write_car_controls(throttle, brake, steering, drs=drs_available)

        # Build state
        state = np.concatenate([
            np.array([normalizedCarPosition, speed, steerAngle, throttle_brake], dtype=np.float32),
            ray_distances
        ])

        # Calculate reward
        reward = self.calculate_reward(state, distance_traveled, steering, throttle_brake, ray_distances)
        
        
        self.total_reward += reward
        self.prev_steering = steering
        self.prev_throttle_brake = throttle_brake

        # Check termination
        done = is_car_off_track(self.car_pos[0], self.car_pos[1], left_polygon, right_polygon)
        
        if done:
            print(f"Total Reward: {self.total_reward}")

        self.prev_state = state
        
        # Rate limiting: ensure each step takes at least step_duration seconds
        elapsed = time.time() - step_start
        if elapsed < self.step_duration:
            time.sleep(self.step_duration - elapsed)
        
        return state, reward, done, False, {}

    def render(self, mode='human'):
        pass

    def calculate_reward(self, state, distance_traveled, steering, throttle_brake, ray_distances):
        pass
      
