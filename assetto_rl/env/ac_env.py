import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .car_control import CarController
from .sim_info import get_car_details, ray_angles
from .track_utils import load_track_data, create_continuous_track_polygon, calculate_ray_endpoint, is_car_off_track


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
        self._prev_laps = None
        self._teleport_on_reset = False

        self.STUCK_SPEED_THRESHOLD = 5.0
        self.STUCK_TIME_STEPS      = 40
        self.STUCK_PENALTY         = -10.0
        self._low_speed_counter    = 0
        self._speed_achieved       = False

        # NEEDED FOR DIRECT MEMORY ACCESS ASSETTO CORSA
        self.INIT_TELEPORT_POS = (490, -11, 248, -0.85, 0, 0.24, 2) # AUSTRIA
        # self.INIT_TELEPORT_POS = (-188.632, 5.346, -254.452, -0.163, 0.000, -0.987, 2) # SPA
        self.RESET_TELEPORT_POS = (0, 0, 0, 1, 0, 0, 0)
        self.INIT_CONTROL = (1, 0, 0)

        # Rate limiting: 20Hz = 1/20 seconds per step
        self.step_duration = 1.0 / 20.0

        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # --- Observation space ---
        base_low  = np.array([
            0,      # normalizedCarPosition
            0,      # speed (km/h)
            -1,     # steerAngle
            -1,     # last_steering
            -1,     # last_throttle_brake
            -25,    # lat_vel
            -5,     # long_vel
            -20,    # yaw_rate
            0,      # avg_slip
            0,      # gas
            0,      # brake
            0,      # drs_available
        ], dtype=np.float32)

        base_high = np.array([
            1,      # normalizedCarPosition
            350,    # speed (km/h)
            1,      # steerAngle
            1,      # last_steering
            1,      # last_throttle_brake
            25,     # lat_vel
            100,    # long_vel
            20,     # yaw_rate
            50,     # avg_slip
            1,      # gas
            1,      # brake
            1,      # drs_available
        ], dtype=np.float32)

        ray_low  = np.zeros(len(ray_angles), dtype=np.float32)
        ray_high = np.ones(len(ray_angles), dtype=np.float32) * 1000

        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, ray_low]),
            high=np.concatenate([base_high, ray_high]),
            dtype=np.float32
        )
        self.obs_size = len(base_low) + len(ray_angles)

    def render(self, mode='human'):
        pass

    def _compute_ray_distances(self, car_pos, heading):
        ray_distances = []
        for angle in ray_angles:
            ray_endpoint = calculate_ray_endpoint(car_pos, heading, angle, left_polygon, right_polygon)
            if ray_endpoint is not None:
                distance = np.linalg.norm(np.array(car_pos) - np.array(ray_endpoint))
                ray_distances.append(min(distance, 1000.0))
            else:
                ray_distances.append(1000.0)
        return np.array(ray_distances, dtype=np.float32)

    def _is_stuck(self, speed):
        if not self._speed_achieved:
            if speed > self.STUCK_SPEED_THRESHOLD:
                self._speed_achieved = True
            return False

        if speed < self.STUCK_SPEED_THRESHOLD:
            self._low_speed_counter += 1
        else:
            self._low_speed_counter = 0

        return self._low_speed_counter >= self.STUCK_TIME_STEPS

    def _build_obs(self, normalizedCarPosition, speed, steerAngle, steering,
                   throttle_brake, lat_vel, long_vel, yaw_rate, avg_slip,
                   gas, brake, drs_available, ray_distances):
        return np.concatenate([
            np.array([
                normalizedCarPosition,
                speed,
                steerAngle,
                steering,
                throttle_brake,
                np.clip(lat_vel,  -25,  25),
                np.clip(long_vel,  -5, 100),
                np.clip(yaw_rate, -20,  20),
                np.clip(avg_slip,   0,  50),
                gas,
                brake,
                float(drs_available),
            ], dtype=np.float32),
            ray_distances
        ])

    def reset(self, seed=None, options=None):
        self.in_lap = False
        self.total_reward = 0.0
        self.prev_steering = 0.0
        self.prev_throttle_brake = 0.0
        self.prev_distance_traveled = None
        self.prev_normalizedCarPosition = None
        self._prev_laps = None
        self._low_speed_counter = 0
        self._speed_achieved = False

        if self._teleport_on_reset:
            # Teleport to initial position
            # ALL OF THESE ARE NEEDED DONT REMOVE ANY OF THEM
            self.car_controller.teleport(*self.INIT_TELEPORT_POS)
            self.car_controller.teleport(*self.RESET_TELEPORT_POS)
            self.car_controller.write_car_controls(*self.INIT_CONTROL)
            # END OF ALL OF THESE ARE NEEDED DONT REMOVE ANY OF THEM

        time.sleep(self.step_duration)
        (car_pos, heading, speed, gas, brake, steerAngle,
         normalizedCarPosition, distance_traveled, laps, drs_available,
         lat_vel, long_vel, yaw_rate, avg_slip) = get_car_details()

        self.car_pos = car_pos
        self.prev_distance_traveled = distance_traveled
        self.prev_normalizedCarPosition = normalizedCarPosition
        self._prev_laps = laps

        ray_distances = self._compute_ray_distances(self.car_pos, heading)

        obs = self._build_obs(
            normalizedCarPosition, speed, steerAngle,
            self.prev_steering, self.prev_throttle_brake,
            lat_vel, long_vel, yaw_rate, avg_slip,
            gas, brake, drs_available, ray_distances
        )
        return obs, {}

    def step(self, action):
        steering = float(action[0])
        throttle_brake = float(action[1])

        throttle = max(0.0, throttle_brake)
        brake_cmd = max(0.0, -throttle_brake)

        # Car state before action
        (car_pos_before, heading_before, speed_before, gas_before, brake_before,
         steerAngle_before, normalizedCarPosition_before, distance_traveled_before,
         laps_before, drs_available, lat_vel, long_vel, yaw_rate, avg_slip) = get_car_details()

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
        (car_pos, heading, speed, gas, brake, steerAngle,
         normalizedCarPosition, distance_traveled, laps, drs_available,
         lat_vel, long_vel, yaw_rate, avg_slip) = get_car_details()

        self.car_pos = car_pos

        # Track lap progress
        if 0 <= normalizedCarPosition < 0.1 and not self.in_lap:
            self.in_lap = True
        if not self.in_lap:
            normalizedCarPosition = 0.0

        # Detect lap completion
        lap_completed = False
        if self._prev_laps is not None and laps > self._prev_laps:
            lap_completed = True
        self._prev_laps = laps

        # Ray distances
        ray_distances = self._compute_ray_distances(self.car_pos, heading)

        # Observation
        state = self._build_obs(
            normalizedCarPosition, speed, steerAngle,
            steering, throttle_brake,
            lat_vel, long_vel, yaw_rate, avg_slip,
            gas, brake, drs_available, ray_distances
        )

        # Termination
        off_track = is_car_off_track(self.car_pos[0], self.car_pos[1], left_polygon, right_polygon)
        stuck = self._is_stuck(speed)
        done = off_track or stuck or lap_completed

        # delta_dist for reward
        if self.prev_distance_traveled is None:
            delta_dist = 0.0
        else:
            delta_dist = max(0.0, distance_traveled - self.prev_distance_traveled)

        reward = self.calculate_reward(
            delta_dist, speed, steering, throttle,
            brake_cmd, throttle_brake,
            off_track, stuck, lap_completed
        )
        self.total_reward += reward
        self.prev_steering = steering
        self.prev_throttle_brake = throttle_brake

        if done:
            if lap_completed:
                print(f"\n[LAP COMPLETE] Total Reward: {self.total_reward:.2f}")
            elif off_track:
                print(f"\n[OFF TRACK] Total Reward: {self.total_reward:.2f}")
            else:
                print(f"\n[STUCK] Total Reward: {self.total_reward:.2f}")

        self._teleport_on_reset = off_track or stuck
        return state, reward, done, False, {}

    def calculate_reward(self, delta_dist, speed, steering, throttle,
                         brake_cmd, throttle_brake,
                         off_track, stuck, lap_completed):
        # --- Weights ---
        progress_weight          = 0.2
        speed_weight             = 0.5
        speed_normalise          = 350.0
        steering_penalty_weight  = 0.15
        steering_change_weight   = 0.5
        throttle_change_weight   = 0.15
        throttle_bonus_weight    = 0.6
        conflict_brake_threshold    = 0.5
        conflict_throttle_threshold = 0.1
        conflict_penalty_value      = -1.5
        off_track_penalty           = -20.0
        lap_completion_bonus        = +20.0
        debug_reward_print          = True

        # Core rewards: progress and speed
        progress_reward = progress_weight * delta_dist / 4
        speed_reward    = speed_weight * (speed / speed_normalise)

        # Universal steering penalty
        steering_penalty = -steering_penalty_weight * (steering ** 2)

        # Unconditional throttle bonus
        throttle_bonus = throttle_bonus_weight * throttle ** 2

        # Smoothness penalties
        steering_change  = steering       - self.prev_steering
        throttle_change  = throttle_brake - self.prev_throttle_brake
        steering_change_penalty = -steering_change_weight  * (steering_change ** 2)
        throttle_change_penalty = -throttle_change_weight * (throttle_change ** 2)

        # Penalise simultaneous throttle + heavy brake
        if brake_cmd > conflict_brake_threshold and throttle > conflict_throttle_threshold:
            conflict_penalty = conflict_penalty_value
        else:
            conflict_penalty = 0.0

        # Terminal rewards/penalties
        exit_penalty         = off_track_penalty    if off_track     else 0.0
        stuck_penalty        = self.STUCK_PENALTY   if stuck         else 0.0
        lap_bonus            = lap_completion_bonus if lap_completed else 0.0

        reward = (
            progress_reward
            + speed_reward
            + steering_penalty
            + throttle_bonus
            + steering_change_penalty
            + throttle_change_penalty
            + conflict_penalty
            + exit_penalty
            + stuck_penalty
            + lap_bonus
        )

        if debug_reward_print:
            projected_total = self.total_reward + reward
            print(
                f"prog={progress_reward:+.3f} "
                f"spd={speed_reward:+.3f} "
                f"str={steering_penalty:+.3f} "
                f"thr_bonus={throttle_bonus:+.3f} "
                f"str_chg={steering_change_penalty:+.3f} "
                f"thr_chg={throttle_change_penalty:+.3f} "
                f"conflict={conflict_penalty:+.3f} "
                f"exit={exit_penalty:+.3f} "
                f"stuck={stuck_penalty:+.3f} "
                f"lap={lap_bonus:+.3f} "
                f"step={reward:+.3f} "
                f"total~={projected_total:+.3f}",
                end="\n",
                flush=True,
            )

        return reward