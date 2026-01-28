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
        self.prev_steering = 0.0
        self.prev_throttle_brake = 0.0
        self.prev_distance_traveled = None
        self.prev_normalizedCarPosition = None
        
        # NEEDED FOR DIRECT MEMORY ACCESS ASETTO CORSA
        self.INIT_TELEPORT_POS = (490, -11, 248, -0.85, 0, 0.24, 2)
        self.RESET_TELEPORT_POS = (0, 0, 0, 1, 0, 0, 0)
        self.INIT_CONTROL = (1, 0, 0)


        # normalizedCarPosition, speed, steerAngle, last_steering, last_throttle_brake + ray distances
        self.OBS_BASE_FEATURES = 5
        
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
        low = np.concatenate([np.array([0, 0, -1, -1, -1], dtype=np.float32), ray_obs_low])
        high = np.concatenate([np.array([1, 350, 1, 1, 1], dtype=np.float32), ray_obs_high])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.obs_size = self.OBS_BASE_FEATURES + len(ray_angles)

    def render(self, mode='human'):
        pass
    
    def _compute_ray_distances(self, car_pos, heading):
        ray_distances = []
        for angle in ray_angles:
            ray_endpoint = calculate_ray_endpoint(car_pos, heading, angle, left_polygon, right_polygon)
            if ray_endpoint is not None:
                distance = np.linalg.norm(np.array(car_pos) - np.array(ray_endpoint))
                ray_distances.append(distance)
            else:
                ray_distances.append(5000)
        return np.array(ray_distances, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.in_lap = False
        self.total_reward = 0.0
        self.prev_steering = 0.0
        self.prev_throttle_brake = 0.0
        self.prev_distance_traveled = None
        self.prev_normalizedCarPosition = None


        # Teleport to initial position
        # ALL OF THESE ARE NEEDED DONT REMOVE ANY OF THEM
        self.car_controller.teleport(*self.INIT_TELEPORT_POS)
        self.car_controller.teleport(*self.RESET_TELEPORT_POS)
        self.car_controller.write_car_controls(*self.INIT_CONTROL)
        # END OF ALL OF THESE ARE NEEDED DONT REMOVE ANY OF THEM

        # Let the sim settle at the reset pose, then return a real initial observation
        time.sleep(self.step_duration)
        car_pos, heading, speed, gas, brake, steerAngle, normalizedCarPosition, distance_traveled, laps, drs_available = get_car_details()
        self.car_pos = car_pos
        self.prev_distance_traveled = distance_traveled
        self.prev_normalizedCarPosition = normalizedCarPosition

        ray_distances = self._compute_ray_distances(self.car_pos, heading)

        reset_state = np.concatenate([
            np.array([normalizedCarPosition, speed, steerAngle, self.prev_steering, self.prev_throttle_brake], dtype=np.float32),
            ray_distances
        ])
        return reset_state, {}

    def step(self, action):
        step_start = time.time()

        steering = float(action[0])
        throttle_brake = float(action[1])

        throttle = max(0.0, throttle_brake)
        brake_cmd = max(0.0, -throttle_brake)

        # car pos before action
        car_pos_before, heading_before, speed_before, gas_before, brake_before, steerAngle_before, normalizedCarPosition_before, distance_traveled_before, laps_before, drs_available = get_car_details()
        self.prev_distance_traveled = distance_traveled_before
        self.prev_normalizedCarPosition = normalizedCarPosition_before

        # Apply controls
        apply_start = time.time()
        self.car_controller.write_car_controls(throttle, brake_cmd, steering, drs=drs_available)

        # 20Hz cool down
        elapsed_after_apply = time.time() - apply_start
        if elapsed_after_apply < self.step_duration:
            time.sleep(self.step_duration - elapsed_after_apply)

        # Get car details after action
        car_pos, heading, speed, gas, brake, steerAngle, normalizedCarPosition, distance_traveled, laps, drs_available = get_car_details()
        self.car_pos = car_pos

        # Track lap progress
        if normalizedCarPosition < 0.1 and normalizedCarPosition >= 0 and not self.in_lap:
            self.in_lap = True
        if not self.in_lap:
            normalizedCarPosition = 0.0

        # Calculate ray distances
        ray_distances = self._compute_ray_distances(self.car_pos, heading)

        # Build observation
        state = np.concatenate([
            np.array([normalizedCarPosition, speed, steerAngle, steering, throttle_brake], dtype=np.float32),
            ray_distances
        ])

        # Calculate reward
        reward = self.calculate_reward(state, distance_traveled, steering, throttle_brake, ray_distances)
        self.total_reward += reward
        self.prev_steering = steering
        self.prev_throttle_brake = throttle_brake

        # Check if car is off track and terminate
        done = is_car_off_track(self.car_pos[0], self.car_pos[1], left_polygon, right_polygon)
        if done:
            print(f"Total Reward: {self.total_reward}")
  
        return state, reward, done, False, {}

    def calculate_reward(self, state, distance_traveled, steering, throttle_brake, ray_distances):
        pass
      
