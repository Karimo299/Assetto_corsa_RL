# from pywinauto import Application
import math
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
        self.total_reward = 0
        self.prev_state = None
        self.prev_laps = 0
        self.in_lap = False
        self.car_pos = [0, 0]
        self.car_controller = CarController()
        # Add low speed tracking variables
        self.low_speed_threshold = 3.0  # 3 km/h
        self.max_low_speed_time_seconds = 60  # Max allowed time at low speed (in seconds)
        self.low_speed_time_seconds = 0.0  # Accumulates time spent at low speed (in seconds)
        # Simulation time step (seconds per step)
        self.time_step = 0.05  # Assuming 20 steps per second, adjust if different

        # --- Tunable parameters (consider moving to config if needed) ---
        self.INIT_TELEPORT_POS = (490, -11, 248, -0.85, 0, 0.24, 2)
        self.RESET_TELEPORT_POS = (0, 0, 0, 1, 0, 0, 0)
        self.INIT_CONTROL = (1, 0, 0)
        self.OBS_BASE_FEATURES = 4  # [track pos, speed, steer, throttle/brake]

        # self.app = Application().connect(title="Assetto Corsa")
        # self.window = self.app.top_window()
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Create observation space for ray distances (0-5000) and other state variables
        ray_obs_low = np.zeros(len(ray_angles), dtype=np.float32)
        ray_obs_high = np.ones(len(ray_angles), dtype=np.float32) * 5000

        # normalized track position, speed, steering angle, throttle/brake, ray distances
        low = np.concatenate([np.array([0, 0, -1, -1], dtype=np.float32), ray_obs_low])
        high = np.concatenate([np.array([1, 350, 1, 1], dtype=np.float32), ray_obs_high])
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
        self.obs_size = self.OBS_BASE_FEATURES + len(ray_angles)

    def reset(self, seed=None, options=None):
        # time.sleep(2)
        print(f"\nResetting environment. Total Reward: {self.total_reward:.2f}\n")
        self.total_reward = 0
        self.in_lap = False
        self.low_speed_time_seconds = 0.0  # Reset low speed timer

        # Teleport to initial position
        self.car_controller.teleport(*self.INIT_TELEPORT_POS)
        # Reset the teleport parameters #DO NOT REMOVE
        self.car_controller.teleport(*self.RESET_TELEPORT_POS)
        self.car_controller.write_car_controls(*self.INIT_CONTROL)
        # time.sleep(5)
        # self.car_controller.write_car_controls(0, 1, 0)

        # Initialize prev_state to correct observation size
        self.prev_state = np.zeros(self.obs_size, dtype=np.float32)
        return self.prev_state, {}

    def step(self, action):
        # Apply the action using the car controller
        steering = float(action[0])  # Between -1 and 1
        throttle_brake = float(action[1])  # Between -1 and 1

        # Split throttle_brake into throttle and brake
        throttle = max(0, throttle_brake)
        brake = max(0, -throttle_brake)

        # Send control signals to the car
        self.car_controller.write_car_controls(throttle, brake, steering)

        # Get new state
        car_pos, heading, speed, gas, brake, steerAngle, normalizedCarPosition, laps = get_car_details()
        self.car_pos = car_pos

        # Track low speed conditions
        if speed < self.low_speed_threshold:
            self.low_speed_time_seconds += self.time_step  # Accumulate seconds
        else:
            self.low_speed_time_seconds = 0.0

        if normalizedCarPosition < 0.1 and normalizedCarPosition >= 0 and not self.in_lap:
            self.in_lap = True

        if not self.in_lap:
            normalizedCarPosition = 0

        ray_distances = []
        for angle in ray_angles:
            # Calculate ray endpoint
            ray_endpoint = calculate_ray_endpoint(self.car_pos, heading, angle, left_polygon, right_polygon)

            # Calculate distance to the track border
            if ray_endpoint is not None:
                distance = np.linalg.norm(np.array(self.car_pos) - np.array(ray_endpoint))
                ray_distances.append(distance)
            else:
                ray_distances.append(5000)

        ray_distances = np.array(ray_distances, dtype=np.float32)

        # Create state vector
        state = np.concatenate([
            np.array([normalizedCarPosition, speed, steerAngle, throttle_brake], dtype=np.float32),
            ray_distances
        ])

        # Calculate reward
        reward = self.calculate_reward(state)

        completed_lap = self.lap_completed(laps)
        print(f"Completed Lap: {completed_lap}, Laps: {laps}, Reward: {reward:.2f}")

        # Check if car is stuck at low speed too long
        stuck_too_long = self.low_speed_time_seconds >= self.max_low_speed_time_seconds
        car_offtrack = is_car_off_track(self.car_pos[0], self.car_pos[1], left_polygon, right_polygon)
        if stuck_too_long:
            print(f"Car stuck at low speed ({speed:.1f} km/h) for {self.low_speed_time_seconds:.1f}s - resetting!")
        if car_offtrack:
            print(f"Car went off track at position ({self.car_pos[0]:.2f}, {self.car_pos[1]:.2f}) - resetting!")

        done = completed_lap or car_offtrack or stuck_too_long

        # Update total reward
        self.total_reward += reward
        # Store current state for next step
        self.prev_state = state

        self.prev_laps = laps

        # Return step information
        return state, reward, done, False, {}  # The fourth value is 'truncated' for gym v0.26+

    def lap_completed(self, laps):
        return laps > self.prev_laps

    def render(self, mode='human'):
        pass

    def calculate_reward(self, state):
        # Extract current state components
        current_progress = state[0]
        speed = state[1]
        steer_angle = state[2]
        throttle_brake = state[3]
        ray_distances = state[4:]

        # Previous state components
        prev_progress = self.prev_state[0] if self.prev_state is not None else 0.0
        prev_steer = self.prev_state[2] if self.prev_state is not None else 0.0
        prev_throttle_brake = self.prev_state[3] if self.prev_state is not None else 0.0

        # 1. Progress Reward (main driver)
        delta_progress = current_progress - prev_progress
        # Heuristic for lap completion wrap-around
        if delta_progress < -0.8:  # Significant backward jump indicates new lap
            delta_progress += 1.0  # Assume full lap progress
        progress_reward = delta_progress * 100.0  # Primary reward component

        # 2. Speed Reward (encourage meaningful speed)
        speed_reward = (speed / 350) * 1.0  # Scaled to max speed

        # 3. Steering Penalty (smooth control)
        steering_penalty = 0.2 * abs(steer_angle) ** 2 #changed from 0.2

        # 4. Action Smoothness Penalty
        delta_steer = abs(steer_angle - prev_steer)
        delta_throttle = abs(throttle_brake - prev_throttle_brake)
        smoothness_penalty = 0.3 * (delta_steer ** 2 + delta_throttle ** 2)
        offtrack_penalty= 0.0
        # 6. Off-track detection (emergency penalty)
        if is_car_off_track(self.car_pos[0], self.car_pos[1], left_polygon, right_polygon):  # All rays maxed out (heuristic)
            offtrack_penalty = 50.0

        # Total reward calculation
        reward = (
            progress_reward
            + speed_reward
            - steering_penalty
            - smoothness_penalty
            - offtrack_penalty
        )

        # Apply additional penalty for very low speeds
        if speed < self.low_speed_threshold:
            # Penalty is proportional to time spent at low speed (in seconds)
            low_speed_penalty = 5.0 * (self.low_speed_time_seconds / self.max_low_speed_time_seconds)
            reward -= low_speed_penalty
            print(f"LowSpeedPen: {-low_speed_penalty:.1f} | ", end="")

        # Diagnostic print
        print(f"[Reward Components] Progress: {progress_reward:.1f} | "
              f"Speed: {speed_reward:.1f} | "
              f"SteerPen: {-steering_penalty:.1f} | "
              f"SmoothPen: {-smoothness_penalty:.1f} | "
              f"offtrack_penalty: {-offtrack_penalty:.1f}")

        return float(reward)

