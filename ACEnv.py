# from pywinauto import Application
import math
import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from CarControl import CarController
from sim_info import get_car_details, ray_angles
from track_utils import load_track_data, create_continuous_track_polygon, calculate_ray_endpoint, is_car_off_track, detect_turn_state


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
        
        # Steering smoothing variables
        self.steering_smoothing_factor = 0.7  # Higher = more smoothing (0.0 = no smoothing, 1.0 = no change)
        self.smoothed_steering = 0.0  # Current smoothed steering value
        
        # Distance-based progress tracking
        self.prev_distance = 0.0  # Previous distance traveled for progress calculation
        
        # Rolling rate measurement
        self.step_times = []
        self.max_step_history = 100  # Keep last 100 step times

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
        self.smoothed_steering = 0.0  # Reset smoothed steering
        self.prev_distance = 0.0  # Reset distance tracking

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
        step_start = time.time()
        
        # Apply the action using the car controller
        steering = float(action[0])  # Between -1 and 1
        throttle_brake = float(action[1])  # Between -1 and 1

        # Apply steering smoothing (exponential moving average)
        self.smoothed_steering = (self.steering_smoothing_factor * self.smoothed_steering + 
                                 (1 - self.steering_smoothing_factor) * steering)

        # Split throttle_brake into throttle and brake
        throttle = max(0, throttle_brake)
        brake = max(0, -throttle_brake)

        # Send control signals to the car (use smoothed steering)
        self.car_controller.write_car_controls(throttle, brake, self.smoothed_steering)

        # Get new state
        car_pos, heading, speed, gas, brake, steerAngle, normalizedCarPosition, distance_traveled, laps, drs_available = get_car_details()
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
        
        # Detect if on straight for potential use in reward shaping
        # Consider both ray-based detection AND DRS availability
        ray_based_straight = detect_turn_state(ray_distances, ray_angles)
        is_straight = ray_based_straight and (drs_available == 1)  # Both conditions must be true
        
        # Auto-activate DRS when available and on straights
        if drs_available == 1 and ray_based_straight:  # Use ray_based_straight for DRS activation
            self.car_controller.write_car_controls(throttle, brake, self.smoothed_steering, drs=True)

        # Create state vector
        state = np.concatenate([
            np.array([normalizedCarPosition, speed, steerAngle, throttle_brake], dtype=np.float32),
            ray_distances
        ])

        # Calculate reward
        reward = self.calculate_reward(state, is_straight, drs_available, ray_based_straight, distance_traveled)

        completed_lap = False
        # completed_lap = self.lap_completed(laps)
        print(f"Completed Lap: {completed_lap}, Laps: {laps}, Reward: {reward:.2f}, Straight: {is_straight}, DRS: {drs_available}")

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
        
        # Calculate step duration and rate
        step_duration = time.time() - step_start
        self.step_times.append(step_duration)
        
        # Keep only recent step times
        if len(self.step_times) > self.max_step_history:
            self.step_times.pop(0)
        
        # Calculate rolling average rate
        if len(self.step_times) > 10:  # Wait for some data
            avg_step_time = sum(self.step_times) / len(self.step_times)
            current_rate = 1.0 / avg_step_time
            print(f"Rolling average rate: {current_rate:.2f} steps/second (avg step time: {avg_step_time:.4f}s)")

        # Return step information
        return state, reward, done, False, {}  # The fourth value is 'truncated' for gym v0.26+

    def lap_completed(self, laps):
        return laps > self.prev_laps

    def render(self, mode='human'):
        pass

    def calculate_reward(self, state, is_straight, drs_available, ray_based_straight, distance_traveled):
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

        # 1. Progress Reward (main driver) - Try multiple methods
        delta_progress = current_progress - prev_progress
        # Heuristic for lap completion wrap-around
        if delta_progress < -0.8:  # Significant backward jump indicates new lap
            delta_progress += 1.0  # Assume full lap progress
            
        # Fallback: Use distance-based progress if normalizedCarPosition is not working
        distance_delta = distance_traveled - self.prev_distance
        if abs(delta_progress) < 0.001 and distance_delta > 0:  # normalizedCarPosition not working, use distance
            delta_progress = distance_delta / 1000.0  # Convert meters to km for progress
            print(f"[PROGRESS FALLBACK] Using distance: {distance_delta:.1f}m = {delta_progress:.4f} progress")
        elif abs(delta_progress) < 0.001 and speed > 5:  # If distance also not working, use speed-based progress
            delta_progress = speed / 10000.0  # Convert speed to progress (very small but non-zero)
            print(f"[PROGRESS FALLBACK] Using speed: {speed:.1f} km/h = {delta_progress:.4f} progress")
        
        progress_reward = delta_progress * 200.0  # MUCH stronger progress incentive
        
        # DEBUG: Print progress values to understand what's happening
        print(f"[PROGRESS DEBUG] Current: {current_progress:.4f}, Prev: {prev_progress:.4f}, Delta: {delta_progress:.4f}, Distance: {distance_traveled:.1f}m, Reward: {progress_reward:.1f}")
        
        # Update distance tracking
        self.prev_distance = distance_traveled

        # 2. Speed Reward (encourage meaningful speed)
        speed_reward = (speed / 350) * 2.0  # Increased speed reward
        
        # 2a. Throttle Reward (encourage acceleration)
        throttle_reward = 0.0
        if throttle_brake > 0:  # Only reward positive throttle
            throttle_reward = throttle_brake * 1.0  # Direct reward for throttle input
        
        # 2b. DRS Reward (encourage using DRS when available)
        drs_reward = 0.0
        if drs_available == 1 and ray_based_straight:  # DRS is available and we're on a straight
            drs_reward = 2.0  # Strong reward for being in DRS zone
        
        # 2b. Stationary Penalty (penalize staying in one place)
        stationary_penalty = 0.0
        if speed < 5.0:  # Very slow or stationary
            stationary_penalty = 10.0  # Strong penalty for not moving

        # 3. Steering Reward/Penalty (quadratic function - rewards small, penalizes large)
        # Only give steering rewards if actually moving (speed > 10 km/h)
        steering_reward_penalty = 0.0
        if speed > 10.0:  # Only reward steering when moving
            if is_straight:
                # On straights: HEAVY penalty for ANY steering
                # Formula: penalty = penalty_coeff * |steering|^power
                penalty_coeff = 10.0  # Much higher penalty coefficient
                power = 1.5  # Between linear and quadratic for aggressive penalty
                steering_penalty = penalty_coeff * (abs(self.smoothed_steering) ** power)
                steering_reward_penalty = -steering_penalty  # Negative = penalty
            else:
                # In turns: much gentler quadratic function
                max_steering_reward = 0.2  # Small reward for zero steering in turns
                penalty_coeff = 0.5  # Gentle penalty increase
                steering_reward_penalty = max_steering_reward - penalty_coeff * (self.smoothed_steering ** 2)
                steering_reward_penalty = max(0.0, steering_reward_penalty)

        # 3b. Braking Penalty (HEAVY penalty for braking on straights)
        braking_penalty = 0.0
        if is_straight and throttle_brake < 0:
            # HEAVY penalty for braking on straights
            if drs_available == 1:
                # EXTRA heavy penalty when DRS is available (should never brake!)
                braking_penalty = 20.0 * abs(throttle_brake)  # Very heavy penalty
            else:
                # Heavy penalty for braking on straights without DRS
                braking_penalty = 10.0 * abs(throttle_brake)  # Heavy penalty

        # 3c. Throttle Penalty (penalize not using full throttle on DRS straights)
        throttle_penalty = 0.0
        if is_straight and drs_available == 1 and throttle_brake < 0.95:
            # Penalize not using near-full throttle when DRS is active
            throttle_penalty = 5.0 * (0.95 - throttle_brake)  # Penalty for not using full throttle

        # 4. Action Smoothness Penalty
        delta_steer = abs(steer_angle - prev_steer)
        delta_throttle = abs(throttle_brake - prev_throttle_brake)
        smoothness_penalty = 0.3 * (delta_steer ** 2 + delta_throttle ** 2)
        
        # 5. Track State Reward (encourage appropriate behavior)
        if is_straight:
            # On straights: reward high speed, full throttle, and minimal steering
            track_state_reward = 0.5  # Small bonus for being on straight
            
            if speed > 250:  # Reward high speed on straights
                track_state_reward += 1.0
            if throttle_brake > 0.95:  # Reward full throttle on straights
                track_state_reward += 0.5
            
            # Steering rewards are now handled by the quadratic function above
        else:
            # In turns: just small bonus for navigating turns (no arbitrary speed)
            track_state_reward = 0.2  # Small bonus for navigating turns
        
        offtrack_penalty= 0.0
        # 6. Off-track detection (emergency penalty)
        if is_car_off_track(self.car_pos[0], self.car_pos[1], left_polygon, right_polygon):  # All rays maxed out (heuristic)
            offtrack_penalty = 500.0

        # Total reward calculation
        reward = (
            progress_reward
            + speed_reward
            + throttle_reward  # New throttle incentive
            + drs_reward  # New DRS incentive
            + track_state_reward
            + steering_reward_penalty  # Can be positive (turns) or negative (straights)
            - braking_penalty
            - throttle_penalty  # New penalty for not using full throttle on DRS straights
            - smoothness_penalty
            - offtrack_penalty
            - stationary_penalty  # New penalty for staying stationary
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
              f"Throttle: {throttle_reward:.1f} | "
              f"DRS: {drs_reward:.1f} | "
              f"TrackState: {track_state_reward:.1f} | "
              f"SteerReward/Pen: {steering_reward_penalty:.1f} | "
              f"BrakePen: {-braking_penalty:.1f} | "
              f"ThrottlePen: {-throttle_penalty:.1f} | "
              f"SmoothPen: {-smoothness_penalty:.1f} | "
              f"OfftrackPen: {-offtrack_penalty:.1f} | "
              f"StationaryPen: {-stationary_penalty:.1f} | "
              f"RawSteer: {steer_angle:.3f} | "
              f"SmoothSteer: {self.smoothed_steering:.3f} | "
              f"ThrottleInput: {throttle_brake:.3f} | "
              f"Straight: {is_straight} | "
              f"DRSAvail: {drs_available} | "
              f"DRSActive: {drs_available == 1 and ray_based_straight}")

        return float(reward)

